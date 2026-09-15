#!/usr/bin/env python3
"""
worker_balance.py: worker-level I/O balance metrics computed from a
simplified JSON produced by simplify_logs.py.

This script never touches a .darshan file. It reads one simplified JSON,
computes per-worker I/O totals, summarizes their balance at run level,
writes CSV files, and optionally generates graphs by reading those CSVs.

Usage:
    python3 worker_balance.py run.json
    python3 worker_balance.py run.json --compact-graphs true
    python3 worker_balance.py run.json --individual-graphs true
    python3 worker_balance.py run.json \
        --compact-graphs true \
        --individual-graphs true
    python3 worker_balance.py run.json -o results/

If --outdir is not given, output is written next to the input JSON:

    <input directory>/worker_balance_out/

Files written:
    proc_metrics.csv   one row per process
    run_metrics.csv    one row per run

Compact graphs:
    graphs/
        compact/
            workers/
                worker_bytes_<run>.png
                worker_reads_<run>.png
                worker_read_time_<run>.png
                worker_balance_summary_<run>.png

Individual graphs:
    graphs/
        individual/
            process/
                worker_profile_<run>_<pid>.png

The analysis focuses only on DataLoader worker processes identified by
simplify_logs.py. It does not use Darshan MPI/rank imbalance counters.

For each worker p:

    B_p = sum(POSIX_BYTES_READ)
    R_p = sum(POSIX_READS)
    T_p = sum(POSIX_F_READ_TIME)

At run level, balance is summarized using:

    mean
    standard deviation
    minimum
    maximum
    coefficient of variation (CV)

where:

    CV = std / mean

The primary balance metrics are:

    worker_bytes_cv
    worker_reads_cv
    worker_read_time_cv

Interpretation:

    bytes CV near 0
        workers read similar amounts of data

    reads CV near 0
        workers perform similar numbers of read operations

    read-time CV near 0
        workers spend similar amounts of time inside POSIX read calls

Useful combinations:

    low bytes CV + low reads CV
        the read workload is distributed fairly uniformly

    low bytes CV + high read-time CV
        workers receive similar data volumes but experience uneven I/O cost

    high reads CV + low bytes CV
        workers move similar data volumes using different numbers of reads,
        potentially indicating different access granularities

Two rules govern every number here:

  * -1 means Darshan could not collect the counter, so it is masked to 0
    before any summation rather than summed as a negative.

  * worker balance is computed directly from the process logs produced by the
    DataLoader workers. Parent-process rows are retained in proc_metrics.csv
    for context but are excluded from worker balance statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
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


def safe_mean(values):
    return statistics.fmean(values) if values else None


def safe_std(values):
    """
    Population standard deviation.

    For one worker, dispersion is zero by definition for this run-level
    descriptive summary.
    """
    if not values:
        return None

    if len(values) == 1:
        return 0.0

    return statistics.pstdev(values)


def describe(values, prefix):
    """
    Return descriptive statistics for one worker-level metric.
    """

    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_cv": None,
            f"{prefix}_max_min_ratio": None,
        }

    mean = safe_mean(values)
    std = safe_std(values)
    minimum = min(values)
    maximum = max(values)

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_cv":
            ratio(std, mean),

        f"{prefix}_max_min_ratio":
            ratio(maximum, minimum),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def file_row(log: dict, rec: dict) -> dict:
    """One record of one log -> one temporary file-level row."""

    c = rec["counters"]
    fc = rec["fcounters"]

    return {
        "run_id": log["run_id"],
        "pid": log["pid"],
        "role": log["role"],
        "file": rec["file"],

        "reads": mask(
            c.get("POSIX_READS", 0)
        ),

        "bytes_read": mask(
            c.get("POSIX_BYTES_READ", 0)
        ),

        "read_time": mask_float(
            fc.get("POSIX_F_READ_TIME", 0.0)
        ),
    }


def summarize_process(rows, **extra) -> dict:
    """
    Collapse one process's file rows into one process-level row.
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

    out.update({
        "n_records": len(rows),
        "n_files": n_files,

        "reads": reads,
        "bytes_read": bytes_read,
        "read_time": read_time,

        "effective_read_bandwidth_MBps":
            (
                bytes_read / 1e6 / read_time
                if read_time
                else None
            ),
    })

    return out


def summarize_run(worker_rows, all_proc_rows, **extra) -> dict:
    """
    Summarize balance across worker processes.

    Parent-process rows are excluded from balance statistics.
    """

    out = dict(extra)

    worker_bytes = [
        row["bytes_read"]
        for row in worker_rows
    ]

    worker_reads = [
        row["reads"]
        for row in worker_rows
    ]

    worker_read_time = [
        row["read_time"]
        for row in worker_rows
    ]

    worker_bandwidth = [
        row["effective_read_bandwidth_MBps"]
        for row in worker_rows
        if row["effective_read_bandwidth_MBps"]
        is not None
    ]

    out.update({
        "n_procs": len(all_proc_rows),
        "n_workers": len(worker_rows),

        "workers_with_reads": sum(
            1
            for row in worker_rows
            if row["reads"] > 0
        ),

        "workers_with_bytes": sum(
            1
            for row in worker_rows
            if row["bytes_read"] > 0
        ),

        "workers_with_read_time": sum(
            1
            for row in worker_rows
            if row["read_time"] > 0
        ),

        "total_worker_reads": sum(
            worker_reads
        ),

        "total_worker_bytes_read": sum(
            worker_bytes
        ),

        "total_worker_read_time": sum(
            worker_read_time
        ),
    })

    out.update(
        describe(
            worker_bytes,
            "worker_bytes",
        )
    )

    out.update(
        describe(
            worker_reads,
            "worker_reads",
        )
    )

    out.update(
        describe(
            worker_read_time,
            "worker_read_time",
        )
    )

    out.update(
        describe(
            worker_bandwidth,
            "worker_bandwidth",
        )
    )

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

def fmt(value, digits=4):
    if value is None:
        return "undefined"

    return f"{value:.{digits}f}"


def print_balance_summary(run_row):
    """
    Compact human-readable run summary.
    """

    print(
        f"    worker bytes CV:     "
        f"{fmt(run_row['worker_bytes_cv'])}"
    )

    print(
        f"    worker reads CV:     "
        f"{fmt(run_row['worker_reads_cv'])}"
    )

    print(
        f"    worker read-time CV: "
        f"{fmt(run_row['worker_read_time_cv'])}"
    )


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
# Compact worker comparison graphs
# ---------------------------------------------------------------------------

def worker_rows_for_run(rows):
    return [
        row
        for row in rows
        if row.get("role") == "worker"
    ]


def graph_worker_metric(
    plt,
    rows,
    output,
    run_id,
    key,
    xlabel,
    title,
    scale=1.0,
):
    """
    Horizontal bars for one worker-level metric.
    """

    workers = worker_rows_for_run(
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
        num(row, key) / scale
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
        xlabel
    )

    ax.set_ylabel(
        "Worker PID"
    )

    ax.set_title(
        title
        + "\n"
        + f"run={run_id}"
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


def graph_worker_balance_summary(
    plt,
    run_row,
    output,
    run_id,
):
    """
    One compact bar chart of the three primary worker balance CVs.
    """

    labels = [
        "Bytes CV",
        "Reads CV",
        "Read-time CV",
    ]

    values = [
        num(
            run_row,
            "worker_bytes_cv",
        ),
        num(
            run_row,
            "worker_reads_cv",
        ),
        num(
            run_row,
            "worker_read_time_cv",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Coefficient of variation"
    )

    ax.set_title(
        "Worker I/O balance summary\n"
        f"run={run_id}"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for i, value in enumerate(values):
        ax.annotate(
            f"{value:.3f}",
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


def graph_compact_workers_csv(
    plt,
    proc_csv,
    run_csv,
    graph_dir,
):
    proc_rows = read_csv(
        proc_csv
    )

    run_rows = read_csv(
        run_csv
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_run = defaultdict(list)

    for row in proc_rows:
        by_run[row["run_id"]].append(row)

    run_summary = {
        row["run_id"]: row
        for row in run_rows
    }

    for run_id, rows in sorted(
        by_run.items()
    ):

        workers = worker_rows_for_run(
            rows
        )

        if len(workers) < 2:
            continue

        graph_worker_metric(
            plt,
            rows,
            graph_dir /
            f"worker_bytes_{run_id}.png",
            run_id,
            "bytes_read",
            "Bytes read (MB)",
            "Bytes read per worker",
            scale=1e6,
        )

        graph_worker_metric(
            plt,
            rows,
            graph_dir /
            f"worker_reads_{run_id}.png",
            run_id,
            "reads",
            "POSIX reads",
            "Read operations per worker",
        )

        graph_worker_metric(
            plt,
            rows,
            graph_dir /
            f"worker_read_time_{run_id}.png",
            run_id,
            "read_time",
            "POSIX read time (s)",
            "Read time per worker",
        )

        if run_id in run_summary:

            graph_worker_balance_summary(
                plt,
                run_summary[run_id],
                graph_dir /
                f"worker_balance_summary_{run_id}.png",
                run_id,
            )


# ---------------------------------------------------------------------------
# Individual process graphs
# ---------------------------------------------------------------------------

def graph_process_profile(
    plt,
    row,
    output,
    run_id,
):
    """
    Small profile of one worker process.

    Values use different units, so each bar is normalized to the run-local
    worker maximum before plotting. Absolute values remain available in the
    CSV and are included in the tick labels.
    """

    bytes_mb = (
        num(row, "bytes_read")
        / 1e6
    )

    reads = num(
        row,
        "reads"
    )

    read_time = num(
        row,
        "read_time"
    )

    values = [
        bytes_mb,
        reads,
        read_time,
    ]

    labels = [
        f"Bytes\n{bytes_mb:.2f} MB",
        f"Reads\n{int(reads):,}",
        f"Read time\n{read_time:.4f} s",
    ]

    nonzero = [
        value
        for value in values
        if value > 0
    ]

    scale = (
        max(nonzero)
        if nonzero
        else 1.0
    )

    normalized = [
        value / scale
        for value in values
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.bar(
        labels,
        normalized,
    )

    ax.set_ylabel(
        "Normalized within process"
    )

    ax.set_ylim(
        0,
        1.1,
    )

    ax.set_title(
        "Worker I/O profile\n"
        f"run={run_id}  pid={row['pid']}"
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


def graph_individual_processes_csv(
    plt,
    proc_csv,
    graph_dir,
):
    rows = read_csv(
        proc_csv
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:

        if row.get("role") != "worker":
            continue

        graph_process_profile(
            plt,
            row,
            graph_dir /
            (
                f"worker_profile_"
                f"{row['run_id']}_"
                f"{row['pid']}.png"
            ),
            row["run_id"],
        )


# ---------------------------------------------------------------------------
# Graph generation entry points
# ---------------------------------------------------------------------------

def generate_compact_graphs(outdir):
    """
    Read generated CSV files and create compact worker-balance summaries.
    """

    plt = setup_matplotlib()

    if plt is None:
        return

    outdir = Path(
        outdir
    )

    graphs = (
        outdir /
        "graphs" /
        "compact" /
        "workers"
    )

    print()
    print("=" * 78)
    print("GENERATING COMPACT WORKER-BALANCE GRAPHS FROM CSV FILES")
    print("=" * 78)

    proc_csv = (
        outdir /
        "proc_metrics.csv"
    )

    run_csv = (
        outdir /
        "run_metrics.csv"
    )

    if (
        proc_csv.exists()
        and run_csv.exists()
    ):

        graph_compact_workers_csv(
            plt,
            proc_csv,
            run_csv,
            graphs,
        )

    print(
        f"\nCompact graphs written to {graphs}/"
    )


def generate_individual_graphs(outdir):
    """
    Read proc_metrics.csv and create one profile graph per worker.
    """

    plt = setup_matplotlib()

    if plt is None:
        return

    outdir = Path(
        outdir
    )

    graphs = (
        outdir /
        "graphs" /
        "individual" /
        "process"
    )

    print()
    print("=" * 78)
    print("GENERATING INDIVIDUAL WORKER GRAPHS FROM CSV FILES")
    print("=" * 78)

    proc_csv = (
        outdir /
        "proc_metrics.csv"
    )

    if proc_csv.exists():

        graph_individual_processes_csv(
            plt,
            proc_csv,
            graphs,
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
            "default: worker_balance_out "
            "next to the input JSON"
        ),
    )

    ap.add_argument(
        "--compact-graphs",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help=(
            "generate compact worker comparison "
            "graphs from the CSV files "
            "(default: false)"
        ),
    )

    ap.add_argument(
        "--individual-graphs",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help=(
            "generate one diagnostic profile graph "
            "per worker process "
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
            / "worker_balance_out"
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
    # Build process-level rows
    # -----------------------------------------------------------------------

    runs = defaultdict(list)

    proc_rows = []

    for log in doc["logs"]:

        file_rows = [
            file_row(
                log,
                rec,
            )
            for rec in log["records"]
        ]

        pr = summarize_process(
            file_rows,
            run_id=log["run_id"],
            pid=log["pid"],
            role=log["role"],
            run_time=log.get("run_time"),
            any_partial=
                log["any_partial"],
            **log["coords"],
        )

        proc_rows.append(
            pr
        )

        runs[
            log["run_id"]
        ].append(
            (
                log,
                pr,
            )
        )


    # -----------------------------------------------------------------------
    # Run-level worker balance
    # -----------------------------------------------------------------------

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
            log["log"]
            for log, _
            in entries
            if log["any_partial"]
        ]

        all_prs = [
            pr
            for _, pr
            in entries
        ]

        worker_prs = [
            pr
            for _, pr
            in entries
            if pr["role"] == "worker"
        ]


        print(
            "=" * 78
        )

        print(
            f"RUN {run_id}   {label}"
        )

        print(
            f"  {len(worker_prs)} workers"
            +
            (
                f"   *** {len(partial)} logs "
                "with incomplete data ***"
                if partial
                else ""
            )
        )


        # ---------------------------------------------------------------
        # Worker details
        # ---------------------------------------------------------------

        for pr in sorted(
            worker_prs,
            key=lambda r: r["pid"],
        ):

            bw = (
                pr["effective_read_bandwidth_MBps"]
            )

            print(
                f"\n  -- worker pid "
                f"{pr['pid']}: "
                f"{pr['reads']:,} reads, "
                f"{pr['bytes_read'] / 1e6:,.2f} MB, "
                f"{pr['read_time']:,.6f} s read time"
                +
                (
                    f", {bw:,.2f} MB/s"
                    if bw is not None
                    else ""
                )
            )


        # ---------------------------------------------------------------
        # Run balance
        # ---------------------------------------------------------------

        agg = summarize_run(
            worker_prs,
            all_prs,
            run_id=run_id,
            n_partial=len(partial),
            **coords,
        )

        run_rows.append(
            agg
        )


        print(
            f"\n  == WORKER BALANCE:"
        )

        print_balance_summary(
            agg
        )

        print()


    # -----------------------------------------------------------------------
    # Write CSV files
    # -----------------------------------------------------------------------

    proc_csv = (
        outdir /
        "proc_metrics.csv"
    )

    run_csv = (
        outdir /
        "run_metrics.csv"
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
