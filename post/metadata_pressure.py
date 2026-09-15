#!/usr/bin/env python3
"""
metadata_pressure.py: metadata-pressure metrics (row 3 of the I/O pattern
table), computed from a simplified JSON produced by simplify_logs.py.

This script never touches a .darshan file. It reads one simplified JSON,
computes metrics at three aggregation levels, writes them as CSV files,
and optionally generates graphs by reading those CSV files.

Usage:
    python3 metadata_pressure.py run.json
    python3 metadata_pressure.py run.json --compact-graphs true
    python3 metadata_pressure.py run.json --individual-graphs true
    python3 metadata_pressure.py run.json \
        --compact-graphs true \
        --individual-graphs true
    python3 metadata_pressure.py run.json -o results/

If --outdir is not given, output is written next to the input JSON:

    <input directory>/metadata_pressure_out/

Files written:
    file_metrics.csv   one row per (run, pid, file)
    proc_metrics.csv   one row per (run, pid)
    run_metrics.csv    one row per run

Compact graphs:
    graphs/
        compact/
            run/
                operation_mix_run_<run>.png
                metadata_burden_run_<run>.png
            workers/
                worker_metadata_ops_<run>.png
                worker_metadata_time_<run>.png
                worker_metadata_variability_<run>.png

Individual graphs:
    graphs/
        individual/
            file/
                operation_mix_file_<run>_<pid>_<file>.png
                metadata_burden_file_<run>_<pid>_<file>.png
            process/
                operation_mix_process_<run>_<pid>.png
                metadata_burden_process_<run>_<pid>.png
            run/
                operation_mix_run_<run>.png
                metadata_burden_run_<run>.png

The operation classes tracked by this analysis are:

    POSIX_READS
    POSIX_WRITES
    POSIX_OPENS
    POSIX_STATS
    POSIX_SEEKS

Metadata operations are defined as:

    metadata_ops = POSIX_OPENS + POSIX_STATS

The denominator used for operation-share metrics is deliberately named
tracked_ops rather than total POSIX operations:

    tracked_ops =
        reads + writes + opens + stats + seeks

so that metadata_ops_pct means:

    100 * metadata_ops / tracked_ops

and does not imply that every possible POSIX operation class is included.

Metadata pressure is described from three complementary viewpoints:

  * operation share:
        metadata_ops_pct
    What fraction of the tracked POSIX operations are metadata operations?

  * metadata intensity relative to useful reads:
        metadata_ops_per_1000_reads
    How much metadata work is performed relative to read activity?

  * metadata time:
        POSIX_F_META_TIME
    How much accumulated process time is spent in metadata operations?

At file/process level, metadata_time_pct_runtime is:

    100 * meta_time / run_time

At run level, summing POSIX_F_META_TIME across concurrently running
processes produces accumulated process-seconds, so:

    aggregate_meta_time / run_time

can exceed 1. It is therefore reported as meta_time_per_runtime and must not
be interpreted as a wall-clock percentage.

For cross-worker comparisons, run_metrics.csv additionally reports:

    meta_time_per_proc_runtime =
        aggregate_meta_time / (run_time * n_procs)

which expresses the average fraction of available process-runtime spent in
metadata operations.

Two rules govern every number here:

  * -1 means Darshan could not collect the counter, so it is masked to 0
    before any summation rather than summed as a negative.

  * ratios are always computed from summed absolute counters. Per-process
    percentages are never averaged to form a run-level percentage.
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

OP_KEYS = [
    "reads",
    "writes",
    "seeks",
    "opens",
    "stats",
]

OP_LABELS = [
    "Reads",
    "Writes",
    "Seeks",
    "Opens",
    "Stats",
]


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


def pct(num, den):
    """Return 100 * num / den, or None when the denominator is zero."""
    return (100.0 * num / den) if den else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def file_row(log: dict, rec: dict) -> dict:
    """One record of one log -> one file-level row."""

    c = rec["counters"]
    fc = rec["fcounters"]

    reads = mask(c.get("POSIX_READS", 0))
    writes = mask(c.get("POSIX_WRITES", 0))
    opens = mask(c.get("POSIX_OPENS", 0))
    stats = mask(c.get("POSIX_STATS", 0))
    seeks = mask(c.get("POSIX_SEEKS", 0))

    metadata_ops = opens + stats
    tracked_ops = (
        reads
        + writes
        + opens
        + stats
        + seeks
    )

    meta_time = mask_float(
        fc.get("POSIX_F_META_TIME", 0.0)
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
        "writes": writes,
        "opens": opens,
        "stats": stats,
        "seeks": seeks,

        "metadata_ops": metadata_ops,
        "tracked_ops": tracked_ops,

        "metadata_ops_fraction":
            ratio(metadata_ops, tracked_ops),

        "metadata_ops_pct":
            pct(metadata_ops, tracked_ops),

        "opens_per_file":
            float(opens),

        "stats_per_file":
            float(stats),

        "metadata_ops_per_file":
            float(metadata_ops),

        "opens_per_1000_reads":
            (1000.0 * opens / reads)
            if reads
            else None,

        "stats_per_1000_reads":
            (1000.0 * stats / reads)
            if reads
            else None,

        "metadata_ops_per_1000_reads":
            (1000.0 * metadata_ops / reads)
            if reads
            else None,

        "meta_time": meta_time,

        "meta_time_per_metadata_op":
            ratio(meta_time, metadata_ops),

        "run_time": run_time,

        "meta_time_per_runtime":
            ratio(meta_time, run_time),

        "metadata_time_pct_runtime":
            pct(meta_time, run_time),
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

    writes = sum(
        r["writes"]
        for r in rows
    )

    opens = sum(
        r["opens"]
        for r in rows
    )

    stats = sum(
        r["stats"]
        for r in rows
    )

    seeks = sum(
        r["seeks"]
        for r in rows
    )

    metadata_ops = opens + stats

    tracked_ops = (
        reads
        + writes
        + opens
        + stats
        + seeks
    )

    meta_time = sum(
        r["meta_time"]
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
        "writes": writes,
        "opens": opens,
        "stats": stats,
        "seeks": seeks,

        "metadata_ops": metadata_ops,
        "tracked_ops": tracked_ops,

        "metadata_ops_fraction":
            ratio(metadata_ops, tracked_ops),

        "metadata_ops_pct":
            pct(metadata_ops, tracked_ops),

        "opens_per_file":
            ratio(opens, n_files),

        "stats_per_file":
            ratio(stats, n_files),

        "metadata_ops_per_file":
            ratio(metadata_ops, n_files),

        "opens_per_1000_reads":
            (1000.0 * opens / reads)
            if reads
            else None,

        "stats_per_1000_reads":
            (1000.0 * stats / reads)
            if reads
            else None,

        "metadata_ops_per_1000_reads":
            (1000.0 * metadata_ops / reads)
            if reads
            else None,

        "meta_time": meta_time,

        "meta_time_per_metadata_op":
            ratio(meta_time, metadata_ops),

        "run_time": run_time,

        "meta_time_per_runtime":
            ratio(meta_time, run_time),

        "metadata_time_pct_runtime":
            pct(meta_time, run_time),
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

def ascii_operations(row, width=44):
    """
    Show the tracked operation composition in the terminal.
    """

    tracked = row["tracked_ops"]

    if not tracked:
        return ["    (no tracked operations)"]

    values = [
        row[key]
        for key in OP_KEYS
    ]

    mx = max(values)

    if not mx:
        return ["    (no tracked operations)"]

    return [
        (
            f"    {label:>10} |"
            f"{'#' * int(round(width * value / mx)):<{width}}| "
            f"{value:>8,}  "
            f"{100 * value / tracked:5.1f}%"
        )
        for label, value
        in zip(OP_LABELS, values)
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


def operation_percentages(row):
    """Return tracked operation classes as percentages of tracked_ops."""
    tracked = num(row, "tracked_ops")

    if not tracked:
        return [0.0] * len(OP_KEYS)

    return [
        100.0 * num(row, key) / tracked
        for key in OP_KEYS
    ]


# ---------------------------------------------------------------------------
# Generic graph functions
# ---------------------------------------------------------------------------

def graph_operation_mix(
    plt,
    row,
    output,
    title,
):
    """
    One 100% horizontal stacked bar for tracked POSIX operations.
    """

    values = operation_percentages(row)

    fig, ax = plt.subplots(
        figsize=(9, 3.2)
    )

    left = 0.0

    for label, value in zip(
        OP_LABELS,
        values,
    ):
        ax.barh(
            [0],
            [value],
            left=left,
            label=label,
        )

        if value >= 5.0:
            ax.text(
                left + value / 2.0,
                0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

        left += value

    ax.set_xlim(
        0,
        100,
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Share of tracked POSIX operations (%)"
    )

    ax.set_title(
        title
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=5,
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


def graph_metadata_burden(
    plt,
    row,
    output,
    title,
    run_level=False,
):
    """
    Show metadata operation share and metadata time normalization.

    At run level, the time bar is accumulated process-time / wall-clock
    runtime and can exceed 100%, so the chart labels it accordingly rather
    than calling it a percentage of wall-clock time.
    """

    metadata_ops_pct = num(
        row,
        "metadata_ops_pct",
    )

    if run_level:
        labels = [
            "Metadata ops\n(% tracked ops)",
            "Meta process-time\n/ runtime (%)",
            "Meta process-time\n/ proc-runtime (%)",
        ]

        values = [
            metadata_ops_pct,
            100.0 * num(
                row,
                "meta_time_per_runtime",
            ),
            100.0 * num(
                row,
                "meta_time_per_proc_runtime",
            ),
        ]
    else:
        labels = [
            "Metadata ops\n(% tracked ops)",
            "Metadata time\n(% runtime)",
        ]

        values = [
            metadata_ops_pct,
            num(
                row,
                "metadata_time_pct_runtime",
            ),
        ]

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
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
            f"{value:.2f}%",
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

        graph_operation_mix(
            plt,
            row,
            graph_dir /
            f"operation_mix_file_{prefix}.png",
            "File tracked-operation mix\n"
            + title_base,
        )

        graph_metadata_burden(
            plt,
            row,
            graph_dir /
            f"metadata_burden_file_{prefix}.png",
            "File metadata burden\n"
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

        graph_operation_mix(
            plt,
            row,
            graph_dir /
            f"operation_mix_process_{prefix}.png",
            "Process tracked-operation mix\n"
            + title_base,
        )

        graph_metadata_burden(
            plt,
            row,
            graph_dir /
            f"metadata_burden_process_{prefix}.png",
            "Process metadata burden\n"
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

        graph_operation_mix(
            plt,
            row,
            graph_dir /
            f"operation_mix_run_{run_id}.png",
            "Run aggregate tracked-operation mix\n"
            f"run={run_id}",
        )

        graph_metadata_burden(
            plt,
            row,
            graph_dir /
            f"metadata_burden_run_{run_id}.png",
            "Run aggregate metadata burden\n"
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

        graph_operation_mix(
            plt,
            row,
            graph_dir /
            f"operation_mix_run_{run_id}.png",
            "Run aggregate tracked-operation mix\n"
            f"run={run_id}",
        )

        graph_metadata_burden(
            plt,
            row,
            graph_dir /
            f"metadata_burden_run_{run_id}.png",
            "Run aggregate metadata burden\n"
            f"run={run_id}",
            run_level=True,
        )


# ---------------------------------------------------------------------------
# Compact worker graphs
# ---------------------------------------------------------------------------

def worker_rows_with_activity(rows):
    """
    Keep worker processes that performed tracked operations or metadata work.
    """
    return [
        row
        for row in rows
        if row.get("role") == "worker"
        and (
            num(row, "tracked_ops") > 0
            or num(row, "meta_time") > 0
        )
    ]


def graph_worker_metadata_ops(
    plt,
    rows,
    output,
    run_id,
):
    """
    Compare metadata operation share across workers.
    """

    workers = worker_rows_with_activity(
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
        num(row, "metadata_ops_pct")
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
        "Metadata operations (% of tracked operations)"
    )

    ax.set_ylabel(
        "Worker PID"
    )

    ax.set_title(
        "Worker metadata-operation share\n"
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


def graph_worker_metadata_time(
    plt,
    rows,
    output,
    run_id,
):
    """
    Compare per-worker metadata time as a percentage of that process runtime.
    """

    workers = worker_rows_with_activity(
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
        num(row, "metadata_time_pct_runtime")
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
        "Metadata time (% of process runtime)"
    )

    ax.set_ylabel(
        "Worker PID"
    )

    ax.set_title(
        "Worker metadata-time share\n"
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


def graph_worker_metadata_variability(
    plt,
    rows,
    output,
    run_id,
):
    """
    Summarize worker-level variability in normalized metadata metrics.

    The plotted metrics have different meanings but share a percentage scale:
      * metadata operations as % of tracked operations
      * metadata time as % of process runtime

    The run-level aggregate remains the authoritative ratio-of-sums result.
    """

    workers = worker_rows_with_activity(
        rows
    )

    if len(workers) < 2:
        return False

    data = [
        [
            num(row, "metadata_ops_pct")
            for row in workers
        ],
        [
            num(row, "metadata_time_pct_runtime")
            for row in workers
        ],
    ]

    labels = [
        "Metadata ops\n(% tracked ops)",
        "Metadata time\n(% runtime)",
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 5)
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_ylabel(
        "Percent"
    )

    ax.set_title(
        "Worker metadata-pressure variability\n"
        f"run={run_id}, n={len(workers)} workers with activity"
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

        graph_worker_metadata_ops(
            plt,
            run_rows,
            graph_dir /
            f"worker_metadata_ops_{run_id}.png",
            run_id,
        )

        graph_worker_metadata_time(
            plt,
            run_rows,
            graph_dir /
            f"worker_metadata_time_{run_id}.png",
            run_id,
        )

        graph_worker_metadata_variability(
            plt,
            run_rows,
            graph_dir /
            f"worker_metadata_variability_{run_id}.png",
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
            "default: metadata_pressure_out "
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
            / "metadata_pressure_out"
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

            metadata_pct = (
                pr["metadata_ops_pct"]
            )

            meta_time_pct = (
                pr["metadata_time_pct_runtime"]
            )

            print(
                f"\n  -- pid "
                f"{pr['pid']} "
                f"({pr['role']}): "
                f"{pr['n_files']} files, "
                f"{pr['tracked_ops']:,} tracked ops, "
                f"{pr['metadata_ops']:,} metadata ops"
                +
                (
                    f" ({metadata_pct:.2f}%)"
                    if metadata_pct is not None
                    else ""
                )
            )

            for line in ascii_operations(
                pr
            ):
                print(line)

            if (
                pr["metadata_ops_per_1000_reads"]
                is not None
            ):
                print(
                    "    metadata ops / 1000 reads: "
                    f"{pr['metadata_ops_per_1000_reads']:,.2f}"
                )

            if meta_time_pct is not None:
                print(
                    "    metadata time / runtime: "
                    f"{meta_time_pct:,.2f}%"
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

        agg["workers_with_metadata"] = sum(
            1
            for pr in prs
            if pr["role"] == "worker"
            and (
                pr["metadata_ops"] > 0
                or pr["meta_time"] > 0
            )
        )

        agg["aggregate_meta_time"] = (
            agg["meta_time"]
        )

        agg["meta_time_per_proc_runtime"] = (
            agg["aggregate_meta_time"]
            / (run_time * n_procs)
            if run_time
            and n_procs
            else None
        )

        agg["metadata_time_pct_proc_runtime"] = (
            100.0
            * agg["meta_time_per_proc_runtime"]
            if agg["meta_time_per_proc_runtime"]
            is not None
            else None
        )


        proc_rows += prs
        run_rows.append(agg)


        metadata_pct = (
            agg["metadata_ops_pct"]
        )

        print(
            f"\n  == AGGREGATE: "
            f"{agg['tracked_ops']:,} tracked ops, "
            f"{agg['metadata_ops']:,} metadata ops"
            +
            (
                f" ({metadata_pct:.2f}%)"
                if metadata_pct is not None
                else ""
            )
        )


        for line in ascii_operations(
            agg
        ):
            print(line)


        if (
            agg["metadata_ops_per_1000_reads"]
            is not None
        ):
            print(
                "    metadata ops / 1000 reads: "
                f"{agg['metadata_ops_per_1000_reads']:,.2f}"
            )


        if (
            agg["meta_time_per_runtime"]
            is not None
        ):
            print(
                "    aggregate meta process-time / runtime: "
                f"{agg['meta_time_per_runtime']:,.4f}"
                " process-seconds per runtime second"
            )


        if (
            agg["meta_time_per_proc_runtime"]
            is not None
        ):
            print(
                "    meta time / available proc-runtime: "
                f"{100 * agg['meta_time_per_proc_runtime']:,.2f}%"
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
