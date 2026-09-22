#!/usr/bin/env python3
"""
simplify_logs.py: read Darshan JSON logs, filter records by name, and
write one simplified JSON.

This script does no metric computation. It only normalises the input so that
downstream scripts never have to know about log formats, PID extraction or
name-record ids.

Usage:

    python3 simplify_logs.py <dir> -i 'mc_normalized_rntuple_10M\\.h5$' -o run.json

    python3 simplify_logs.py <dir> -i '\\.h5$' -e '__pycache__' -o all_h5.json

    python3 simplify_logs.py <dir> -o everything.json          # no filter

Input format:

    *.darshan.json -> read from `python -m darshan to_json` output

Output schema (see SCHEMA below): a top-level object with the filter that was
applied and a list of logs. Each log carries its role, run grouping, module
partial flags, experiment coordinates parsed from the command line, and the
records that survived the filter with counters already named.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SCHEMA = "darshan-simplified/1"


# Experiment coordinates are parsed from the command line Darshan recorded,
# not from the directory name, so a renamed directory cannot mislabel a run.
EXE_ARGS = {
    "fmt": r"(\w+)_tr\.py",
    "strategy": r"--strategy[= ]+(\S+)",
    "model": r"--model[= ]+(\S+)",
    "workers": r"--num-workers[= ]+(\d+)",
    "epochs": r"--epochs[= ]+(\d+)",
    "batch": r"--batch-size[= ]+(\d+)",
    "data_path": r"--data-path[= ]+(\S+)",
}


def coords_from_exe(exe: str) -> dict:
    out = {}

    for key, pat in EXE_ARGS.items():
        m = re.search(pat, exe or "")
        out[key] = m.group(1) if m else None

    return out


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_json_log(path: Path) -> dict:
    """Log exported with `python -m darshan to_json`."""

    with open(path) as fh:
        d = json.load(fh)

    names = d.get("name_records", {})

    cn = (
        d.get("counters", {})
        .get("POSIX", {})
        .get("counters", [])
    )

    fn = (
        d.get("counters", {})
        .get("POSIX", {})
        .get("fcounters", [])
    )

    raw = (
        d.get("records", {})
        .get("POSIX", [])
    )

    # e.g. DXT_POSIX can come back as a string
    if not isinstance(raw, list):
        raw = []

    recs = [
        {
            "file": names.get(str(r["id"]), f"<id:{r['id']}>"),
            "rank": r.get("rank"),
            "counters": dict(zip(cn, r["counters"])),
            "fcounters": dict(zip(fn, r["fcounters"])),
        }
        for r in raw
    ]

    return _pack(
        path,
        d["metadata"]["job"],
        d.get("metadata", {}).get("exe", ""),
        d.get("modules", {}),
        recs,
    )


def _pack(
    path: Path,
    job: dict,
    exe: str,
    modules: dict,
    recs: list,
) -> dict:
    """Common structure.

    The PID pair comes from the filename, which Darshan writes as

        <user>_<binary>_id<PPID>-<PID>_<date>_<uniqueid>_<timing>.darshan.json

    PPID groups the logs of one run; fork_parent in the job metadata confirms
    that a process is a fork rather than the parent.
    """

    m = re.search(r"_id(\d+)-(\d+)_", path.name)

    ppid = int(m.group(1)) if m else job.get("jobid")
    pid = int(m.group(2)) if m else job.get("jobid")

    fork_parent = (job.get("metadata") or {}).get("fork_parent")

    role = (
        "worker"
        if fork_parent
        else ("parent" if pid == ppid else "worker")
    )

    partial = {
        k: bool(v.get("partial_flag"))
        for k, v in modules.items()
    }

    return {
        "log": path.name,
        "path": str(path),
        "run_id": str(ppid),
        "pid": pid,
        "role": role,
        "fork_parent": fork_parent,
        "jobid": job.get("jobid"),
        "nprocs": job.get("nprocs"),
        "run_time": job.get("run_time"),
        "start_time_sec": job.get("start_time_sec"),
        "end_time_sec": job.get("end_time_sec"),
        "exe": exe,
        "coords": coords_from_exe(exe),
        "modules": list(modules),
        "partial_flag": partial,
        "any_partial": any(partial.values()),
        "records": recs,
    }


def load(path: Path) -> dict | None:
    try:
        return read_json_log(path)

    except Exception as e:
        # one bad log must not kill the whole batch
        print(
            f"  WARNING: could not read {path.name}: {e}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "logdir",
        help="directory holding the logs (searched recursively)",
    )

    ap.add_argument(
        "--include",
        "-i",
        default=None,
        help="keep only records whose name matches this regex",
    )

    ap.add_argument(
        "--exclude",
        "-e",
        default=None,
        help=(
            "drop records whose name matches this regex "
            "(applied after include)"
        ),
    )

    ap.add_argument(
        "--output",
        "-o",
        default="simplified.json",
    )

    args = ap.parse_args()

    # Only operate on JSON files produced from Darshan logs.
    logs = sorted(
        Path(args.logdir).rglob("*.darshan.json")
    )

    if not logs:
        sys.exit(
            f"No *.darshan.json found under {args.logdir}"
        )

    inc = re.compile(args.include) if args.include else None
    exc = re.compile(args.exclude) if args.exclude else None

    print(f"{len(logs)} logs found")
    print(f"  include: {args.include or '(none)'}")
    print(f"  exclude: {args.exclude or '(none)'}\n")

    out_logs = []
    n_kept = 0
    n_total = 0
    n_partial = 0

    for p in logs:
        log = load(p)

        if log is None:
            continue

        recs = log["records"]

        kept = [
            r
            for r in recs
            if (not inc or inc.search(r["file"]))
            and not (exc and exc.search(r["file"]))
        ]

        if inc and not kept:
            print(
                f"  WARNING: filter matched nothing in {p.name}",
                file=sys.stderr,
            )

        log["n_records_total"] = len(recs)
        log["n_records_kept"] = len(kept)
        log["records"] = kept

        n_total += len(recs)
        n_kept += len(kept)
        n_partial += bool(log["any_partial"])

        out_logs.append(log)

        flag = (
            "  *** INCOMPLETE ***"
            if log["any_partial"]
            else ""
        )

        print(
            f"  {log['role']:<6} "
            f"pid {log['pid']:<9} "
            f"run {log['run_id']:<9} "
            f"{len(kept):>5}/{len(recs):<6} records"
            f"{flag}"
        )

    doc = {
        "schema": SCHEMA,
        "source_dir": str(Path(args.logdir).resolve()),
        "include": args.include,
        "exclude": args.exclude,
        "n_logs": len(out_logs),
        "n_runs": len({l["run_id"] for l in out_logs}),
        "logs": out_logs,
    }

    with open(args.output, "w") as fh:
        json.dump(doc, fh, indent=1)

    print(
        f"\n{len(out_logs)} logs, "
        f"{doc['n_runs']} runs, "
        f"{n_kept:,}/{n_total:,} records kept"
        + (
            f", {n_partial} logs with incomplete data"
            if n_partial
            else ""
        )
    )

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()