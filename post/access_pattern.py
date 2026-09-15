#!/usr/bin/env python3
"""
access_pattern.py: sequentiality and fragmentation metrics (row 2 of the
I/O pattern table), computed from a simplified JSON produced by
simplify_logs.py.

This script never touches a .darshan file. It reads one simplified JSON,
computes metrics at three aggregation levels, writes them as CSV files,
and optionally generates graphs by reading those CSV files.

Usage:
    python3 access_pattern.py run.json
    python3 access_pattern.py run.json --compact-graphs true
    python3 access_pattern.py run.json --individual-graphs true
    python3 access_pattern.py run.json \
        --compact-graphs true \
        --individual-graphs true
    python3 access_pattern.py run.json -o results/

If --outdir is not given, output is written next to the input JSON:

    <input directory>/access_pattern_out/

Files written:
    file_metrics.csv   one row per (run, pid, file)
    proc_metrics.csv   one row per (run, pid)
    run_metrics.csv    one row per run

Compact graphs:
    graphs/
        compact/
            run/
                pattern_run_<run>.png
            workers/
                worker_pattern_<run>.png
                worker_variability_<run>.png

Individual graphs:
    graphs/
        individual/
            file/
                pattern_file_<run>_<pid>_<file>.png
                seeks_file_<run>_<pid>_<file>.png
            process/
                pattern_process_<run>_<pid>.png
                seeks_process_<run>_<pid>.png
            run/
                pattern_run_<run>.png
                seeks_run_<run>.png

The access pattern is split into three mutually exclusive categories:

    consecutive
        POSIX_CONSEC_READS

    sequential_with_gaps
        POSIX_SEQ_READS - POSIX_CONSEC_READS

    nonsequential
        POSIX_READS - POSIX_SEQ_READS

The corresponding fractions are computed against POSIX_READS.

Two rules govern every number here:

  * -1 means Darshan could not collect the counter, so it is masked to 0
    before any summation rather than summed as a negative.

  * ratios are always computed from summed absolute counters. Per-process
    percentages are never averaged to form a run-level percentage, because
    that would give a process with 20 reads the same weight as one with a
    million reads.

POSIX_SEEKS is kept as an auxiliary signal. It measures explicit seek
operations and is therefore not treated as a direct definition of
fragmentation. The normalized form reported here is seeks per 1000 reads.
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

PATTERN_KEYS = [
    "consecutive_reads",
    "sequential_gap_reads",
    "nonsequential_reads",
]

PATTERN_LABELS = [
    "Consecutive",
    "Sequential with gaps",
    "Non-sequential",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask(v):
    """-1 -> 0. Applied before any summation."""
    return 0 if v == INVALID else v


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


def normalized_pattern(reads, seq_reads, consec_reads):
    """
    Convert Darshan counters into mutually exclusive read categories.

    Darshan's intended relationship is:

        consecutive <= sequential <= reads

    Defensive clamping prevents malformed or partially collected counters
    from producing negative derived categories.
    """
    reads = max(0, reads)
    seq_reads = min(max(0, seq_reads), reads)
    consec_reads = min(max(0, consec_reads), seq_reads)

    sequential_gap_reads = seq_reads - consec_reads
    nonsequential_reads = reads - seq_reads

    return (
        consec_reads,
        sequential_gap_reads,
        nonsequential_reads,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def file_row(log: dict, rec: dict) -> dict:
    """One record of one log -> one file-level row."""

    c = rec["counters"]

    reads = mask(c.get("POSIX_READS", 0))
    seq_reads = mask(c.get("POSIX_SEQ_READS", 0))
    consec_reads = mask(c.get("POSIX_CONSEC_READS", 0))
    seeks = mask(c.get("POSIX_SEEKS", 0))

    (
        consecutive,
        sequential_gap,
        nonsequential,
    ) = normalized_pattern(
        reads,
        seq_reads,
        consec_reads,
    )

    return {
        "run_id": log["run_id"],
        "pid": log["pid"],
        "role": log["role"],
        "file": rec["file"],

        "reads": reads,
        "seq_reads": seq_reads,
        "consec_reads": consec_reads,
        "seeks": seeks,

        "consecutive_reads": consecutive,
        "sequential_gap_reads": sequential_gap,
        "nonsequential_reads": nonsequential,

        "consecutive_ratio":
            ratio(consecutive, reads),

        "sequential_gap_ratio":
            ratio(sequential_gap, reads),

        "nonsequential_ratio":
            ratio(nonsequential, reads),

        "sequential_ratio":
            ratio(seq_reads, reads),

        "seeks_per_read":
            ratio(seeks, reads),

        "seeks_per_1000_reads":
            (1000.0 * seeks / reads)
            if reads
            else None,
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

    seq_reads = sum(
        r["seq_reads"]
        for r in rows
    )

    consec_reads = sum(
        r["consec_reads"]
        for r in rows
    )

    seeks = sum(
        r["seeks"]
        for r in rows
    )

    (
        consecutive,
        sequential_gap,
        nonsequential,
    ) = normalized_pattern(
        reads,
        seq_reads,
        consec_reads,
    )

    out = dict(extra)

    out.update({
        "n_rows": len(rows),

        "reads": reads,
        "seq_reads": seq_reads,
        "consec_reads": consec_reads,
        "seeks": seeks,

        "consecutive_reads": consecutive,
        "sequential_gap_reads": sequential_gap,
        "nonsequential_reads": nonsequential,

        "consecutive_ratio":
            ratio(consecutive, reads),

        "sequential_gap_ratio":
            ratio(sequential_gap, reads),

        "nonsequential_ratio":
            ratio(nonsequential, reads),

        "sequential_ratio":
            ratio(seq_reads, reads),

        "seeks_per_read":
            ratio(seeks, reads),

        "seeks_per_1000_reads":
            (1000.0 * seeks / reads)
            if reads
            else None,
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

def ascii_pattern(row, width=44):
    """
    Show a compact three-part read-pattern summary in the terminal.
    """

    reads = row["reads"]

    if not reads:
        return ["    (no reads)"]

    values = [
        row[key]
        for key in PATTERN_KEYS
    ]

    return [
        (
            f"    {label:>20} |"
            f"{'#' * int(round(width * value / reads)):<{width}}| "
            f"{value:>8,}  "
            f"{100 * value / reads:5.1f}%"
        )
        for label, value
        in zip(PATTERN_LABELS, values)
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


def pattern_percentages(row):
    """Return the three mutually exclusive read categories in percent."""
    reads = num(row, "reads")

    if not reads:
        return [0.0, 0.0, 0.0]

    return [
        100.0 * num(row, key) / reads
        for key in PATTERN_KEYS
    ]


# ---------------------------------------------------------------------------
# Generic graph functions
# ---------------------------------------------------------------------------

def graph_pattern_bar(
    plt,
    row,
    output,
    title,
):
    """
    One 100% horizontal stacked bar showing the read access pattern.
    """

    values = pattern_percentages(row)

    fig, ax = plt.subplots(
        figsize=(8, 3.0)
    )

    left = 0.0

    for label, value in zip(
        PATTERN_LABELS,
        values,
    ):
        ax.barh(
            [0],
            [value],
            left=left,
            label=label,
        )

        if value >= 4.0:
            ax.text(
                left + value / 2.0,
                0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )

        left += value

    ax.set_xlim(
        0,
        100,
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Share of POSIX reads (%)"
    )

    ax.set_title(
        title
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
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


def graph_seek_activity(
    plt,
    row,
    output,
    title,
):
    """
    Normalized explicit seek activity for one row.
    """

    value = num(
        row,
        "seeks_per_1000_reads",
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.bar(
        ["Seeks / 1000 reads"],
        [value],
    )

    ax.set_ylabel(
        "Explicit seek operations"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.annotate(
        f"{value:,.2f}",
        (0, value),
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

        graph_pattern_bar(
            plt,
            row,
            graph_dir /
            f"pattern_file_{prefix}.png",
            "File read access pattern\n"
            + title_base,
        )

        graph_seek_activity(
            plt,
            row,
            graph_dir /
            f"seeks_file_{prefix}.png",
            "File seek activity\n"
            + title_base,
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

        graph_pattern_bar(
            plt,
            row,
            graph_dir /
            f"pattern_process_{prefix}.png",
            "Process read access pattern\n"
            + title_base,
        )

        graph_seek_activity(
            plt,
            row,
            graph_dir /
            f"seeks_process_{prefix}.png",
            "Process seek activity\n"
            + title_base,
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

        graph_pattern_bar(
            plt,
            row,
            graph_dir /
            f"pattern_run_{run_id}.png",
            "Run aggregate read access pattern\n"
            f"run={run_id}",
        )

        graph_seek_activity(
            plt,
            row,
            graph_dir /
            f"seeks_run_{run_id}.png",
            "Run aggregate seek activity\n"
            f"run={run_id}",
        )


# ---------------------------------------------------------------------------
# Compact run graph
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

        graph_pattern_bar(
            plt,
            row,
            graph_dir /
            f"pattern_run_{run_id}.png",
            "Run aggregate read access pattern\n"
            f"run={run_id}",
        )


# ---------------------------------------------------------------------------
# Compact worker comparison graphs
# ---------------------------------------------------------------------------

def graph_worker_pattern(
    plt,
    rows,
    output,
    run_id,
):
    """
    One 100% stacked horizontal bar per worker.

    Only worker processes with at least one read are included.
    """

    worker_rows = [
        row
        for row in rows
        if row.get("role") == "worker"
        and num(row, "reads") > 0
    ]

    if len(worker_rows) < 2:
        return False

    worker_rows.sort(
        key=lambda r: int(r["pid"])
    )

    pids = [
        str(row["pid"])
        for row in worker_rows
    ]

    values_by_category = [
        [
            pattern_percentages(row)[i]
            for row in worker_rows
        ]
        for i in range(3)
    ]

    height = max(
        4.0,
        min(
            24.0,
            1.8 + 0.28 * len(worker_rows),
        ),
    )

    fig, ax = plt.subplots(
        figsize=(10, height)
    )

    left = [
        0.0
        for _ in worker_rows
    ]

    y = list(
        range(len(worker_rows))
    )

    for label, values in zip(
        PATTERN_LABELS,
        values_by_category,
    ):
        ax.barh(
            y,
            values,
            left=left,
            label=label,
        )

        left = [
            a + b
            for a, b
            in zip(left, values)
        ]

    ax.set_xlim(
        0,
        100,
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
        "Share of POSIX reads (%)"
    )

    ax.set_ylabel(
        "Worker PID"
    )

    ax.set_title(
        "Worker read access patterns\n"
        f"run={run_id}"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
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


def graph_worker_variability(
    plt,
    rows,
    output,
    run_id,
):
    """
    Boxplots of worker-level access-pattern percentages.

    This graph summarizes variability between workers. It does not replace
    the run-level ratio-of-sums metrics stored in run_metrics.csv.
    """

    worker_rows = [
        row
        for row in rows
        if row.get("role") == "worker"
        and num(row, "reads") > 0
    ]

    if len(worker_rows) < 2:
        return False

    data = [
        [
            pattern_percentages(row)[i]
            for row in worker_rows
        ]
        for i in range(3)
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        data,
        tick_labels=PATTERN_LABELS,
        showmeans=True,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Worker-level share of POSIX reads (%)"
    )

    ax.set_title(
        "Worker access-pattern variability\n"
        f"run={run_id}, n={len(worker_rows)} workers with reads"
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

        graph_worker_pattern(
            plt,
            run_rows,
            graph_dir /
            f"worker_pattern_{run_id}.png",
            run_id,
        )

        graph_worker_variability(
            plt,
            run_rows,
            graph_dir /
            f"worker_variability_{run_id}.png",
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
            "default: access_pattern_out "
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
            / "access_pattern_out"
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

            pr["n_files"] = (
                pr.pop("n_rows")
            )

            prs.append(pr)

            print(
                f"\n  -- pid "
                f"{pr['pid']} "
                f"({pr['role']}): "
                f"{pr['n_files']} files, "
                f"{pr['reads']:,} reads, "
                f"{pr['seq_reads']:,} sequential, "
                f"{pr['consec_reads']:,} consecutive, "
                f"{pr['seeks']:,} seeks"
            )

            for line in ascii_pattern(
                pr
            ):
                print(line)

            seeks_norm = (
                pr["seeks_per_1000_reads"]
            )

            if seeks_norm is not None:
                print(
                    f"    seeks / 1000 reads: "
                    f"{seeks_norm:,.2f}"
                )


        # ---------------------------------------------------------------
        # Run level
        # ---------------------------------------------------------------

        all_rows = [
            row
            for _, rows in entries
            for row in rows
        ]

        agg = summarize(
            all_rows,
            run_id=run_id,
            n_procs=len(entries),
            n_workers=sum(
                1
                for log, _
                in entries
                if log["role"] == "worker"
            ),
            n_partial=len(partial),
            **coords,
        )

        agg["n_records"] = (
            agg.pop("n_rows")
        )

        agg["n_files"] = len(
            {
                r["file"]
                for r in all_rows
            }
        )

        agg["workers_with_reads"] = sum(
            1
            for pr in prs
            if pr["role"] == "worker"
            and pr["reads"] > 0
        )


        proc_rows += prs
        run_rows.append(agg)


        print(
            f"\n  == AGGREGATE: "
            f"{agg['reads']:,} reads, "
            f"{agg['seq_reads']:,} sequential, "
            f"{agg['consec_reads']:,} consecutive, "
            f"{agg['seeks']:,} seeks"
        )


        for line in ascii_pattern(
            agg
        ):
            print(line)


        seeks_norm = (
            agg["seeks_per_1000_reads"]
        )

        if seeks_norm is not None:
            print(
                f"    seeks / 1000 reads: "
                f"{seeks_norm:,.2f}"
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
