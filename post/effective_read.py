#!/usr/bin/env python3
"""
effective_bandwidth.py: effective read-bandwidth metrics (row 4 of the
I/O pattern table), computed from a simplified JSON produced by
simplify_logs.py.

This script never touches a .darshan file. It reads one simplified JSON,
computes metrics at three aggregation levels, writes them as CSV files,
and optionally generates graphs by reading those CSV files.

Usage:
    python3 effective_bandwidth.py run.json
    python3 effective_bandwidth.py run.json --compact-graphs true
    python3 effective_bandwidth.py run.json --individual-graphs true
    python3 effective_bandwidth.py run.json \
        --compact-graphs true \
        --individual-graphs true
    python3 effective_bandwidth.py run.json -o results/

If --outdir is not given, output is written next to the input JSON:

    <input directory>/effective_bandwidth_out/

Files written:
    file_metrics.csv   one row per (run, pid, file)
    proc_metrics.csv   one row per (run, pid)
    run_metrics.csv    one row per run

Compact graphs:
    graphs/
        compact/
            run/
                bandwidth_run_<run>.png
                read_time_run_<run>.png
            workers/
                worker_bandwidth_<run>.png
                worker_bandwidth_variability_<run>.png
                worker_read_time_<run>.png

Individual graphs:
    graphs/
        individual/
            file/
                bandwidth_file_<run>_<pid>_<file>.png
                read_time_file_<run>_<pid>_<file>.png
            process/
                bandwidth_process_<run>_<pid>.png
                read_time_process_<run>_<pid>.png
            run/
                bandwidth_run_<run>.png
                read_time_run_<run>.png

The primary per-file and per-process metric is:

    effective_read_bandwidth_MBps =
        POSIX_BYTES_READ / POSIX_F_READ_TIME / 1e6

This measures transfer efficiency during time spent in read operations.

At run level, two different bandwidth concepts are reported:

    aggregate_effective_bandwidth_MBps =
        sum(POSIX_BYTES_READ) / sum(POSIX_F_READ_TIME) / 1e6

and:

    run_throughput_MBps =
        sum(POSIX_BYTES_READ) / run_time / 1e6

The first is a ratio of accumulated bytes to accumulated process read-time.
The second measures total bytes read per second of observed run wall time and
is the more direct scaling metric for comparing different worker counts.

As with the metadata script, summed process read-time can exceed wall-clock
runtime when workers read concurrently. Therefore:

    aggregate_read_time_per_runtime =
        sum(read_time) / run_time

may exceed 1, while:

    read_time_per_proc_runtime =
        sum(read_time) / (run_time * n_procs)

normalizes accumulated read-time by available process-runtime.

Two rules govern every number here:

  * -1 means Darshan could not collect the counter, so it is masked to 0
    before any summation rather than summed as a negative.

  * ratios are always computed from summed absolute counters. Per-process
    bandwidth values are never averaged to form the run aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


INVALID = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask(v):
    """-1 -> 0. Applied before any summation."""
    return 0 if v == INVALID else v


def mask_float(v):
    """-1 -> 0.0 for floating-point counters."""
    return 0.0 if v == INVALID else float(v or 0.0)


def parse_bool(value):
    """
    argparse boolean parser.

    Accepted:
        true / false
        yes / no
        1 / 0
    """
    value = str(value).lower()

    if value in ("true", "yes", "1"):
        return True

    if value in ("false", "no", "0"):
        return False

    raise argparse.ArgumentTypeError(
        "expected true or false"
    )


def safe_name(value):
    """Convert arbitrary text into a filesystem-safe name."""
    value = str(value)
    value = os.path.basename(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ratio(num, den):
    """Return num / den, or None when the denominator is zero."""
    return (num / den) if den else None


def mbps(bytes_read, seconds):
    """Decimal MB/s from bytes and seconds."""
    return (
        bytes_read / 1e6 / seconds
        if seconds
        else None
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def file_row(log: dict, rec: dict) -> dict:
    """One record of one log -> one file-level row."""

    c = rec["counters"]
    fc = rec["fcounters"]

    bytes_read = mask(
        c.get("POSIX_BYTES_READ", 0)
    )

    reads = mask(
        c.get("POSIX_READS", 0)
    )

    read_time = mask_float(
        fc.get("POSIX_F_READ_TIME", 0.0)
    )

    run_time = log.get("run_time")
    run_time = (
        float(run_time)
        if run_time not in (None, "")
        else None
    )

    return {
        "run_id": log["run_id"],
        "pid": log["pid"],
        "role": log["role"],
        "file": rec["file"],

        "reads": reads,
        "bytes_read": bytes_read,
        "read_time": read_time,
        "run_time": run_time,

        "effective_read_bandwidth_MBps":
            mbps(bytes_read, read_time),

        "read_time_per_runtime":
            ratio(read_time, run_time),

        "read_time_pct_runtime":
            (
                100.0 * read_time / run_time
                if run_time
                else None
            ),
    }


def summarize(rows, **extra) -> dict:
    """
    Collapse file-level rows into one process- or run-level row.

    Ratios are always ratios of sums.
    """

    reads = sum(
        r["reads"]
        for r in rows
    )

    bytes_read = sum(
        r["bytes_read"]
        for r in rows
    )

    read_time = sum(
        r["read_time"]
        for r in rows
    )

    n_files = len(
        {
            r["file"]
            for r in rows
        }
    )

    out = dict(extra)

    run_time = out.get("run_time")
    run_time = (
        float(run_time)
        if run_time not in (None, "")
        else None
    )

    out.update({
        "n_rows": len(rows),
        "n_files": n_files,

        "reads": reads,
        "bytes_read": bytes_read,
        "read_time": read_time,
        "run_time": run_time,

        "effective_read_bandwidth_MBps":
            mbps(bytes_read, read_time),

        "read_time_per_runtime":
            ratio(read_time, run_time),

        "read_time_pct_runtime":
            (
                100.0 * read_time / run_time
                if run_time
                else None
            ),
    })

    return out


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows, path):
    if not rows:
        return

    cols = list(rows[0])

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=cols,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def ascii_bandwidth(row, width=44):
    """
    Show a compact bandwidth/read-time summary in the terminal.
    """

    bw = row.get(
        "effective_read_bandwidth_MBps"
    )

    if bw is None:
        bw_text = "undefined"
    else:
        bw_text = f"{bw:,.2f} MB/s"

    return [
        f"    bytes read: {row['bytes_read']:,}",
        f"    read time:  {row['read_time']:,.6f} s",
        f"    bandwidth:  {bw_text}",
    ]


# ===========================================================================
# GRAPHING FROM CSV FILES
# ===========================================================================

def read_csv(path):
    """Read a metrics CSV as dictionaries."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    """
    Safely convert a CSV value to float.

    Empty values such as None mean the metric was undefined.
    """

    value = row.get(key)

    if value in (None, "", "None"):
        return 0.0

    try:
        return float(value)

    except (TypeError, ValueError):
        return 0.0


def setup_matplotlib():
    """
    Import matplotlib only when graphs were requested.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        return plt

    except ImportError:
        print(
            "WARNING: matplotlib is not installed; "
            "CSV files were written but graphs cannot be generated.",
            file=sys.stderr,
        )

        return None


# ---------------------------------------------------------------------------
# Generic graph functions
# ---------------------------------------------------------------------------

def graph_bandwidth(
    plt,
    row,
    output,
    title,
    run_level=False,
):
    """
    Bandwidth summary for one CSV row.
    """

    if run_level:
        labels = [
            "Aggregate effective\nbandwidth",
            "Run throughput",
        ]

        values = [
            num(
                row,
                "aggregate_effective_bandwidth_MBps",
            ),
            num(
                row,
                "run_throughput_MBps",
            ),
        ]
    else:
        labels = [
            "Effective read\nbandwidth"
        ]

        values = [
            num(
                row,
                "effective_read_bandwidth_MBps",
            )
        ]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "MB/s"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for i, value in enumerate(values):
        ax.annotate(
            f"{value:,.2f}",
            (i, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
    )

    plt.close(fig)


def graph_read_time(
    plt,
    row,
    output,
    title,
    run_level=False,
):
    """
    Read-time normalization for one CSV row.
    """

    if run_level:
        labels = [
            "Read process-time\n/ runtime (%)",
            "Read process-time\n/ proc-runtime (%)",
        ]

        values = [
            100.0 * num(
                row,
                "aggregate_read_time_per_runtime",
            ),
            100.0 * num(
                row,
                "read_time_per_proc_runtime",
            ),
        ]
    else:
        labels = [
            "Read time\n(% runtime)"
        ]

        values = [
            num(
                row,
                "read_time_pct_runtime",
            )
        ]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Percent"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for i, value in enumerate(values):
        ax.annotate(
            f"{value:,.2f}%",
            (i, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual file-level graphs
# ---------------------------------------------------------------------------

def graph_file_csv(
    plt,
    csv_path,
    graph_dir,
):
    rows = read_csv(csv_path)

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:

        run_id = row["run_id"]
        pid = row["pid"]
        file_name = safe_name(
            row["file"]
        )

        prefix = (
            f"{run_id}_{pid}_{file_name}"
        )

        title_base = (
            f"run={run_id}  "
            f"pid={pid}\n"
            f"{row['file']}"
        )

        graph_bandwidth(
            plt,
            row,
            graph_dir /
            f"bandwidth_file_{prefix}.png",
            "File effective read bandwidth\n"
            + title_base,
            run_level=False,
        )

        graph_read_time(
            plt,
            row,
            graph_dir /
            f"read_time_file_{prefix}.png",
            "File read-time share\n"
            + title_base,
            run_level=False,
        )


# ---------------------------------------------------------------------------
# Individual process-level graphs
# ---------------------------------------------------------------------------

def graph_process_csv(
    plt,
    csv_path,
    graph_dir,
):
    rows = read_csv(csv_path)

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:

        run_id = row["run_id"]
        pid = row["pid"]
        role = row.get(
            "role",
            "",
        )

        prefix = (
            f"{run_id}_{pid}"
        )

        title_base = (
            f"run={run_id}  "
            f"pid={pid}  "
            f"role={role}"
        )

        graph_bandwidth(
            plt,
            row,
            graph_dir /
            f"bandwidth_process_{prefix}.png",
            "Process effective read bandwidth\n"
            + title_base,
            run_level=False,
        )

        graph_read_time(
            plt,
            row,
            graph_dir /
            f"read_time_process_{prefix}.png",
            "Process read-time share\n"
            + title_base,
            run_level=False,
        )


# ---------------------------------------------------------------------------
# Individual run-level graphs
# ---------------------------------------------------------------------------

def graph_run_csv(
    plt,
    csv_path,
    graph_dir,
):
    rows = read_csv(csv_path)

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:

        run_id = row["run_id"]

        graph_bandwidth(
            plt,
            row,
            graph_dir /
            f"bandwidth_run_{run_id}.png",
            "Run read bandwidth\n"
            f"run={run_id}",
            run_level=True,
        )

        graph_read_time(
            plt,
            row,
            graph_dir /
            f"read_time_run_{run_id}.png",
            "Run aggregate read-time usage\n"
            f"run={run_id}",
            run_level=True,
        )


# ---------------------------------------------------------------------------
# Compact run graphs
# ---------------------------------------------------------------------------

def graph_compact_run_csv(
    plt,
    csv_path,
    graph_dir,
):
    rows = read_csv(csv_path)

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:

        run_id = row["run_id"]

        graph_bandwidth(
            plt,
            row,
            graph_dir /
            f"bandwidth_run_{run_id}.png",
            "Run read bandwidth\n"
            f"run={run_id}",
            run_level=True,
        )

        graph_read_time(
            plt,
            row,
            graph_dir /
            f"read_time_run_{run_id}.png",
            "Run aggregate read-time usage\n"
            f"run={run_id}",
            run_level=True,
        )


# ---------------------------------------------------------------------------
# Compact worker graphs
# ---------------------------------------------------------------------------

def worker_rows_with_reads(rows):
    """
    Keep worker processes that performed reads or accumulated read time.
    """
    return [
        row
        for row in rows
        if row.get("role") == "worker"
        and (
            num(row, "bytes_read") > 0
            or num(row, "read_time") > 0
        )
    ]


def graph_worker_bandwidth(
    plt,
    rows,
    output,
    run_id,
):
    """
    Compare effective read bandwidth across workers.
    """

    workers = worker_rows_with_reads(
        rows
    )

    if len(workers) < 2:
        return False

    workers.sort(
        key=lambda r: int(r["pid"])
    )

    pids = [
        str(row["pid"])
        for row in workers
    ]

    values = [
        num(
            row,
            "effective_read_bandwidth_MBps",
        )
        for row in workers
    ]

    height = max(
        4.0,
        min(
            24.0,
            1.8 + 0.28 * len(workers),
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, height)
    )

    y = list(
        range(len(workers))
    )

    ax.barh(
        y,
        values,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        pids,
        fontsize=8,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Effective read bandwidth (MB/s)"
    )

    ax.set_ylabel(
        "Worker PID"
    )

    ax.set_title(
        "Worker effective read bandwidth\n"
        f"run={run_id}"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


def graph_worker_bandwidth_variability(
    plt,
    rows,
    output,
    run_id,
):
    """
    Boxplot of worker-level effective read bandwidth.
    """

    workers = worker_rows_with_reads(
        rows
    )

    if len(workers) < 2:
        return False

    values = [
        num(
            row,
            "effective_read_bandwidth_MBps",
        )
        for row in workers
    ]

    fig, ax = plt.subplots(
        figsize=(6.5, 4.8)
    )

    ax.boxplot(
        [values],
        tick_labels=[
            "Effective read bandwidth"
        ],
        showmeans=True,
    )

    ax.set_ylabel(
        "MB/s"
    )

    ax.set_title(
        "Worker bandwidth variability\n"
        f"run={run_id}, n={len(workers)} workers with reads"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
    )

    plt.close(fig)

    return True


def graph_worker_read_time(
    plt,
    rows,
    output,
    run_id,
):
    """
    Compare worker read-time share of process runtime.
    """

    workers = worker_rows_with_reads(
        rows
    )

    if len(workers) < 2:
        return False

    workers.sort(
        key=lambda r: int(r["pid"])
    )

    pids = [
        str(row["pid"])
        for row in workers
    ]

    values = [
        num(
            row,
            "read_time_pct_runtime",
        )
        for row in workers
    ]

    height = max(
        4.0,
        min(
            24.0,
            1.8 + 0.28 * len(workers),
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, height)
    )

    y = list(
        range(len(workers))
    )

    ax.barh(
        y,
        values,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        pids,
        fontsize=8,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Read time (% of process runtime)"
    )

    ax.set_ylabel(
        "Worker PID"
    )

    ax.set_title(
        "Worker read-time share\n"
        f"run={run_id}"
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


def graph_compact_workers_csv(
    plt,
    csv_path,
    graph_dir,
):
    rows = read_csv(csv_path)

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_run = defaultdict(list)

    for row in rows:
        by_run[row["run_id"]].append(row)

    for run_id, run_rows in sorted(
        by_run.items()
    ):

        graph_worker_bandwidth(
            plt,
            run_rows,
            graph_dir /
            f"worker_bandwidth_{run_id}.png",
            run_id,
        )

        graph_worker_bandwidth_variability(
            plt,
            run_rows,
            graph_dir /
            f"worker_bandwidth_variability_{run_id}.png",
            run_id,
        )

        graph_worker_read_time(
            plt,
            run_rows,
            graph_dir /
            f"worker_read_time_{run_id}.png",
            run_id,
        )


# ---------------------------------------------------------------------------
# Graph generation entry points
# ---------------------------------------------------------------------------

def generate_compact_graphs(outdir):
    """
    Read generated CSV files and create compact run/worker summaries.
    """

    plt = setup_matplotlib()

    if plt is None:
        return

    outdir = Path(outdir)

    graphs = (
        outdir /
        "graphs" /
        "compact"
    )

    print()
    print("=" * 78)
    print("GENERATING COMPACT GRAPHS FROM CSV FILES")
    print("=" * 78)

    run_csv = (
        outdir /
        "run_metrics.csv"
    )

    proc_csv = (
        outdir /
        "proc_metrics.csv"
    )

    if run_csv.exists():

        print(
            "  aggregate run graphs..."
        )

        graph_compact_run_csv(
            plt,
            run_csv,
            graphs / "run",
        )

    if proc_csv.exists():

        print(
            "  compact worker comparison graphs..."
        )

        graph_compact_workers_csv(
            plt,
            proc_csv,
            graphs / "workers",
        )

    print(
        f"\nCompact graphs written to {graphs}/"
    )


def generate_individual_graphs(outdir):
    """
    Read generated CSV files and create file/process/run diagnostic graphs.
    """

    plt = setup_matplotlib()

    if plt is None:
        return

    outdir = Path(outdir)

    graphs = (
        outdir /
        "graphs" /
        "individual"
    )

    print()
    print("=" * 78)
    print("GENERATING INDIVIDUAL GRAPHS FROM CSV FILES")
    print("=" * 78)

    file_csv = (
        outdir /
        "file_metrics.csv"
    )

    proc_csv = (
        outdir /
        "proc_metrics.csv"
    )

    run_csv = (
        outdir /
        "run_metrics.csv"
    )

    if file_csv.exists():

        print(
            "  file-level graphs..."
        )

        graph_file_csv(
            plt,
            file_csv,
            graphs / "file",
        )

    if proc_csv.exists():

        print(
            "  process-level graphs..."
        )

        graph_process_csv(
            plt,
            proc_csv,
            graphs / "process",
        )

    if run_csv.exists():

        print(
            "  run-level graphs..."
        )

        graph_run_csv(
            plt,
            run_csv,
            graphs / "run",
        )

    print(
        f"\nIndividual graphs written to {graphs}/"
    )


# ===========================================================================
# Main
# ===========================================================================

def main():

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=
            argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "simplified",
        help=(
            "JSON produced by "
            "simplify_logs.py"
        ),
    )

    ap.add_argument(
        "--outdir",
        "-o",
        default=None,
        help=(
            "output directory; "
            "default: effective_bandwidth_out "
            "next to the input JSON"
        ),
    )

    ap.add_argument(
        "--compact-graphs",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help=(
            "generate compact run and worker "
            "comparison graphs from the CSV files "
            "(default: false)"
        ),
    )

    ap.add_argument(
        "--individual-graphs",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help=(
            "generate all file-, process-, and run-level "
            "diagnostic graphs from the CSV files "
            "(default: false)"
        ),
    )

    args = ap.parse_args()


    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------

    if args.outdir is None:

        args.outdir = str(
            Path(args.simplified)
            .resolve()
            .parent
            / "effective_bandwidth_out"
        )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------------------------
    # Read simplified JSON
    # -----------------------------------------------------------------------

    with open(
        args.simplified
    ) as fh:

        doc = json.load(fh)


    if (
        doc.get(
            "schema",
            "",
        ).split("/")[0]
        != "darshan-simplified"
    ):

        sys.exit(
            f"{args.simplified} "
            "is not a simplify_logs.py output"
        )


    print(
        f"{doc['n_logs']} logs, "
        f"{doc['n_runs']} runs"
    )

    print(
        "  filter applied at simplify time: "
        f"include={doc.get('include')!r} "
        f"exclude={doc.get('exclude')!r}"
    )

    print(
        f"  output directory: {outdir}"
    )

    print(
        f"  compact graphs: {args.compact_graphs}"
    )

    print(
        f"  individual graphs: {args.individual_graphs}\n"
    )


    # -----------------------------------------------------------------------
    # File-level metrics
    # -----------------------------------------------------------------------

    runs = defaultdict(list)

    file_rows = []

    for log in doc["logs"]:

        rows = [
            file_row(log, rec)
            for rec in log["records"]
        ]

        file_rows += rows

        runs[
            log["run_id"]
        ].append(
            (log, rows)
        )


    # -----------------------------------------------------------------------
    # Process + run metrics
    # -----------------------------------------------------------------------

    proc_rows = []
    run_rows = []

    for run_id, entries in sorted(
        runs.items()
    ):

        coords = (
            entries[0][0]["coords"]
        )

        label = "  ".join(
            f"{k}={v}"
            for k, v
            in coords.items()
            if v
            and k != "data_path"
        )

        partial = [
            entry[0]["log"]
            for entry in entries
            if entry[0]["any_partial"]
        ]

        run_time_candidates = [
            entry[0].get("run_time")
            for entry in entries
            if entry[0].get("run_time")
            not in (None, "")
        ]

        run_time = (
            max(
                float(v)
                for v in run_time_candidates
            )
            if run_time_candidates
            else None
        )


        print(
            "=" * 78
        )

        print(
            f"RUN {run_id}   {label}"
        )

        print(
            f"  {len(entries)} logs"
            +
            (
                f"   *** {len(partial)} "
                "with incomplete data ***"
                if partial
                else ""
            )
        )


        # ---------------------------------------------------------------
        # Process level
        # ---------------------------------------------------------------

        prs = []

        for log, rows in sorted(
            entries,
            key=lambda e: (
                e[0]["role"]
                != "parent",
                e[0]["pid"],
            ),
        ):

            pr = summarize(
                rows,
                run_id=run_id,
                pid=log["pid"],
                role=log["role"],
                run_time=log["run_time"],
                any_partial=
                    log["any_partial"],
                **coords,
            )

            pr["n_records"] = (
                pr.pop("n_rows")
            )

            prs.append(pr)

            bw = (
                pr["effective_read_bandwidth_MBps"]
            )

            print(
                f"\n  -- pid "
                f"{pr['pid']} "
                f"({pr['role']}): "
                f"{pr['n_files']} files, "
                f"{pr['reads']:,} reads, "
                f"{pr['bytes_read'] / 1e6:,.2f} MB"
                +
                (
                    f", {bw:,.2f} MB/s"
                    if bw is not None
                    else ""
                )
            )

            for line in ascii_bandwidth(
                pr
            ):
                print(line)

            if (
                pr["read_time_pct_runtime"]
                is not None
            ):
                print(
                    "    read time / runtime: "
                    f"{pr['read_time_pct_runtime']:,.2f}%"
                )


        # ---------------------------------------------------------------
        # Run level
        # ---------------------------------------------------------------

        all_rows = [
            row
            for _, rows in entries
            for row in rows
        ]

        n_procs = len(entries)

        agg = summarize(
            all_rows,
            run_id=run_id,
            n_procs=n_procs,
            n_workers=sum(
                1
                for log, _
                in entries
                if log["role"] == "worker"
            ),
            n_partial=len(partial),
            run_time=run_time,
            **coords,
        )

        agg["n_records"] = (
            agg.pop("n_rows")
        )

        agg["workers_with_reads"] = sum(
            1
            for pr in prs
            if pr["role"] == "worker"
            and (
                pr["bytes_read"] > 0
                or pr["read_time"] > 0
            )
        )

        agg[
            "aggregate_effective_bandwidth_MBps"
        ] = (
            mbps(
                agg["bytes_read"],
                agg["read_time"],
            )
        )

        agg[
            "run_throughput_MBps"
        ] = (
            mbps(
                agg["bytes_read"],
                run_time,
            )
        )

        agg[
            "aggregate_read_time_per_runtime"
        ] = (
            ratio(
                agg["read_time"],
                run_time,
            )
        )

        agg[
            "read_time_per_proc_runtime"
        ] = (
            agg["read_time"]
            / (run_time * n_procs)
            if run_time
            and n_procs
            else None
        )

        agg[
            "read_time_pct_proc_runtime"
        ] = (
            100.0
            * agg["read_time_per_proc_runtime"]
            if agg["read_time_per_proc_runtime"]
            is not None
            else None
        )


        proc_rows += prs
        run_rows.append(agg)


        print(
            f"\n  == AGGREGATE: "
            f"{agg['reads']:,} reads, "
            f"{agg['bytes_read'] / 1e6:,.2f} MB"
        )


        if (
            agg["aggregate_effective_bandwidth_MBps"]
            is not None
        ):
            print(
                "    aggregate effective bandwidth: "
                f"{agg['aggregate_effective_bandwidth_MBps']:,.2f} MB/s"
            )


        if (
            agg["run_throughput_MBps"]
            is not None
        ):
            print(
                "    run throughput: "
                f"{agg['run_throughput_MBps']:,.2f} MB/s"
            )


        if (
            agg["aggregate_read_time_per_runtime"]
            is not None
        ):
            print(
                "    aggregate read process-time / runtime: "
                f"{agg['aggregate_read_time_per_runtime']:,.4f}"
                " process-seconds per runtime second"
            )


        if (
            agg["read_time_per_proc_runtime"]
            is not None
        ):
            print(
                "    read time / available proc-runtime: "
                f"{100 * agg['read_time_per_proc_runtime']:,.2f}%"
            )

        print()


    # -----------------------------------------------------------------------
    # Write CSV files
    # -----------------------------------------------------------------------

    file_csv = (
        outdir /
        "file_metrics.csv"
    )

    proc_csv = (
        outdir /
        "proc_metrics.csv"
    )

    run_csv = (
        outdir /
        "run_metrics.csv"
    )


    write_csv(
        file_rows,
        file_csv,
    )

    write_csv(
        proc_rows,
        proc_csv,
    )

    write_csv(
        run_rows,
        run_csv,
    )


    print(
        "=" * 78
    )

    print(
        f"Written to {outdir}/:"
    )

    print(
        "  file_metrics.csv"
    )

    print(
        "  proc_metrics.csv"
    )

    print(
        "  run_metrics.csv"
    )


    # -----------------------------------------------------------------------
    # Optional graph generation
    # -----------------------------------------------------------------------

    if args.compact_graphs:
        generate_compact_graphs(
            outdir
        )

    if args.individual_graphs:
        generate_individual_graphs(
            outdir
        )


if __name__ == "__main__":
    main()
