"""
io_intensity.py: access-size histograms (row 1 of the I/O pattern
table), computed from a simplified JSON produced by simplify_logs.py.

This script never touches a .darshan file. It reads one simplified JSON,
computes metrics at three aggregation levels, writes them as CSV files,
and optionally generates graphs by reading those CSV files.

Usage:
    python3 io_intensity.py run.json
    python3 io_intensity.py run.json --graphs true
    python3 io_intensity.py run.json --graphs false
    python3 io_intensity.py run.json -o results/ --graphs true

If --outdir is not given, output is written next to the input JSON:

    <input directory>/io_intensity_out/

Files written:
    file_metrics.csv   one row per (run, pid, file)
    proc_metrics.csv   one row per (run, pid)
    run_metrics.csv    one row per run

When --graphs true:

    graphs/
        file/
            hist_file_<run>_<pid>_<file>.png
            ops_file_<run>_<pid>_<file>.png

        process/
            hist_process_<run>_<pid>.png
            ops_process_<run>_<pid>.png

        run/
            hist_run_<run>.png
            ops_run_<run>.png

Two rules govern every number here:

  * -1 means Darshan could not collect the counter, so it is masked to 0
    before any summation rather than summed as a negative.

  * histogram counts are summed in absolute terms and normalised afterwards.
    Averaging per-process percentages would weight a process that issued 20
    reads the same as one that issued a million.
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
# Darshan read-size histogram
# ---------------------------------------------------------------------------

READ_BUCKETS = [
    "POSIX_SIZE_READ_0_100",
    "POSIX_SIZE_READ_100_1K",
    "POSIX_SIZE_READ_1K_10K",
    "POSIX_SIZE_READ_10K_100K",
    "POSIX_SIZE_READ_100K_1M",
    "POSIX_SIZE_READ_1M_4M",
    "POSIX_SIZE_READ_4M_10M",
    "POSIX_SIZE_READ_10M_100M",
    "POSIX_SIZE_READ_100M_1G",
    "POSIX_SIZE_READ_1G_PLUS",
]

SHORT = [
    b.replace("POSIX_SIZE_READ_", "")
    for b in READ_BUCKETS
]

BUCKET_HI = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    4_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    float("inf"),
]

SMALL_CUTOFF = 10_000


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def file_row(log: dict, rec: dict) -> dict:
    """One record of one log -> one row."""

    c = rec["counters"]
    fc = rec["fcounters"]

    reads = mask(c.get("POSIX_READS", 0))
    bytes_r = mask(c.get("POSIX_BYTES_READ", 0))

    hist = [
        mask(c.get(bucket, 0))
        for bucket in READ_BUCKETS
    ]

    tot = sum(hist)

    small = sum(
        h
        for h, hi in zip(hist, BUCKET_HI)
        if hi <= SMALL_CUTOFF
    )

    row = {
        "run_id": log["run_id"],
        "pid": log["pid"],
        "role": log["role"],
        "file": rec["file"],

        "reads": reads,
        "bytes_read": bytes_r,

        "opens": mask(c.get("POSIX_OPENS", 0)),
        "seeks": mask(c.get("POSIX_SEEKS", 0)),
        "stats": mask(c.get("POSIX_STATS", 0)),

        "max_byte_read": mask(
            c.get("POSIX_MAX_BYTE_READ", 0)
        ),

        "read_time": fc.get(
            "POSIX_F_READ_TIME", 0.0
        ),

        "meta_time": fc.get(
            "POSIX_F_META_TIME", 0.0
        ),

        "hist_total": tot,

        "mean_read_size":
            (bytes_r / reads)
            if reads
            else None,

        "frac_reads_lt_10k":
            (small / tot)
            if tot
            else None,
    }

    row.update(
        dict(zip(SHORT, hist))
    )

    return row


def summarize(rows, **extra) -> dict:
    """
    Collapse rows into one.

    Ratios are always computed as ratios of sums.
    """

    hist = [
        sum(r[s] for r in rows)
        for s in SHORT
    ]

    tot = sum(hist)

    reads = sum(
        r["reads"]
        for r in rows
    )

    bytes_r = sum(
        r["bytes_read"]
        for r in rows
    )

    small = sum(
        h
        for h, hi in zip(hist, BUCKET_HI)
        if hi <= SMALL_CUTOFF
    )

    out = dict(extra)

    out.update({
        "n_rows": len(rows),

        "reads": reads,
        "bytes_read": bytes_r,

        "opens": sum(
            r["opens"]
            for r in rows
        ),

        "seeks": sum(
            r["seeks"]
            for r in rows
        ),

        "stats": sum(
            r["stats"]
            for r in rows
        ),

        "read_time": sum(
            r["read_time"]
            for r in rows
        ),

        "meta_time": sum(
            r["meta_time"]
            for r in rows
        ),

        "hist_total": tot,

        "mean_read_size":
            (bytes_r / reads)
            if reads
            else None,

        "frac_reads_lt_10k":
            (small / tot)
            if tot
            else None,
    })

    out.update(
        dict(zip(SHORT, hist))
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
# Terminal histogram
# ---------------------------------------------------------------------------

def ascii_hist(hist, width=44):

    tot = sum(hist)

    if not tot:
        return ["    (no reads)"]

    mx = max(hist)

    return [
        (
            f"    {label:>10} |"
            f"{'#' * int(round(width * n / mx)):<{width}}| "
            f"{n:>8,}  "
            f"{100 * n / tot:5.1f}%"
        )
        for label, n
        in zip(SHORT, hist)
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

def graph_histogram(
    plt,
    row,
    output,
    title,
):
    """
    Access-size histogram from one CSV row.
    """

    values = [
        num(row, bucket)
        for bucket in SHORT
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        range(len(SHORT)),
        values,
    )

    ax.set_xticks(
        range(len(SHORT))
    )

    ax.set_xticklabels(
        SHORT,
        rotation=55,
        ha="right",
    )

    ax.set_xlabel(
        "Read access size"
    )

    ax.set_ylabel(
        "Number of reads"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for i, value in enumerate(values):

        if value:
            ax.annotate(
                f"{int(value):,}",
                (i, value),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
    )

    plt.close(fig)


def graph_operations(
    plt,
    row,
    output,
    title,
):
    """
    Basic POSIX operation counts available in all three CSV levels.
    """

    labels = [
        "reads",
        "opens",
        "seeks",
        "stats",
    ]

    values = [
        num(row, key)
        for key in labels
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Operation count"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for i, value in enumerate(values):

        if value:
            ax.annotate(
                f"{int(value):,}",
                (i, value),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=150,
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# File-level graphs
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

        graph_histogram(
            plt,
            row,
            graph_dir /
            f"hist_file_{prefix}.png",
            "File access-size histogram\n"
            + title_base,
        )

        graph_operations(
            plt,
            row,
            graph_dir /
            f"ops_file_{prefix}.png",
            "File POSIX operations\n"
            + title_base,
        )


# ---------------------------------------------------------------------------
# Process-level graphs
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

        graph_histogram(
            plt,
            row,
            graph_dir /
            f"hist_process_{prefix}.png",
            "Process access-size histogram\n"
            + title_base,
        )

        graph_operations(
            plt,
            row,
            graph_dir /
            f"ops_process_{prefix}.png",
            "Process POSIX operations\n"
            + title_base,
        )


# ---------------------------------------------------------------------------
# Run-level graphs
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

        graph_histogram(
            plt,
            row,
            graph_dir /
            f"hist_run_{run_id}.png",
            f"Run aggregate access-size histogram\n"
            f"run={run_id}",
        )

        graph_operations(
            plt,
            row,
            graph_dir /
            f"ops_run_{run_id}.png",
            f"Run aggregate POSIX operations\n"
            f"run={run_id}",
        )


# ---------------------------------------------------------------------------
# All graphs
# ---------------------------------------------------------------------------

def generate_graphs(outdir):
    """
    Read the generated CSV files and create all graphs.
    """

    plt = setup_matplotlib()

    if plt is None:
        return

    outdir = Path(outdir)

    graphs = (
        outdir /
        "graphs"
    )

    print()
    print("=" * 78)
    print("GENERATING GRAPHS FROM CSV FILES")
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
        f"\nGraphs written to {graphs}/"
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
            "default: io_intensity_out "
            "next to the input JSON"
        ),
    )

    ap.add_argument(
        "--graphs",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help=(
            "generate graphs from the "
            "resulting CSV files "
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
            / "io_intensity_out"
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
        f"  graphs: {args.graphs}\n"
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

            mrs = (
                pr["mean_read_size"]
            )

            print(
                f"\n  -- pid "
                f"{pr['pid']} "
                f"({pr['role']}): "
                f"{pr['n_files']} files, "
                f"{pr['reads']:,} reads, "
                f"{pr['bytes_read'] / 1e6:,.1f} MB"
                +
                (
                    f", mean size "
                    f"{mrs:,.0f} B"
                    if mrs
                    else ""
                )
            )

            for line in ascii_hist(
                [
                    pr[s]
                    for s in SHORT
                ]
            ):
                print(line)


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


        proc_rows += prs
        run_rows.append(agg)


        mrs = (
            agg["mean_read_size"]
        )

        frac = (
            agg["frac_reads_lt_10k"]
        )


        print(
            f"\n  == AGGREGATE: "
            f"{agg['reads']:,} reads, "
            f"{agg['bytes_read'] / 1e6:,.1f} MB"
            +
            (
                f", mean size "
                f"{mrs:,.0f} B"
                if mrs
                else ""
            )
            +
            (
                f", {100 * frac:.1f}% "
                "under 10 kB"
                if frac is not None
                else ""
            )
        )


        for line in ascii_hist(
            [
                agg[s]
                for s in SHORT
            ]
        ):
            print(line)

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

    if args.graphs:
        generate_graphs(
            outdir
        )


if __name__ == "__main__":
    main()