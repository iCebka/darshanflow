#!/usr/bin/env python3
"""
darshan_readable.py

Converts a JSON file exported by Darshan (darshan-parser --json / pydarshan)
into a human-readable report that associates each file (from "name_records")
with the I/O operations recorded on it (from "records"), using the counter
names defined in "counters".

Usage:
    python3 darshan_readable.py file.darshan.json
    python3 darshan_readable.py file.darshan.json --json output.json
    python3 darshan_readable.py file.darshan.json --all   (also show zero-valued counters)

Expected structure of the input JSON:
    - name_records: { "numeric_id": "path/to/file", ... }
    - counters:     { "MODULE": {"counters": [...names...], "fcounters": [...names...]} }
    - records:      { "MODULE": [ {"id": ..., "rank": ..., "counters": [...], "fcounters": [...]}, ... ] }
"""

import json
import sys
import argparse
from collections import OrderedDict


def load_darshan_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_id_to_name(data):
    """Numeric id (int) -> file name/path map, built from name_records."""
    name_records = data.get("name_records", {})
    return {int(k): v for k, v in name_records.items()}


def build_readable_records(data, include_zero=False):
    """
    Walks through each module inside 'records' and builds, for every record,
    a dictionary {counter_name: value} using the names defined in 'counters'
    for that module.

    Returns a dict: { module: [ {file, rank, id, counters:{...}, fcounters:{...}}, ... ] }
    """
    id_to_name = build_id_to_name(data)
    counters_def = data.get("counters", {})
    records = data.get("records", {})

    result = OrderedDict()

    for module, recs in records.items():
        # Some modules (e.g. DXT_POSIX when it's not enabled) come as an
        # informational string instead of a list of records.
        if not isinstance(recs, list):
            result[module] = {"info": recs}
            continue

        module_def = counters_def.get(module, {})
        counter_names = module_def.get("counters", [])
        fcounter_names = module_def.get("fcounters", [])

        module_result = []
        for rec in recs:
            rec_id = rec.get("id")
            filename = id_to_name.get(rec_id, f"<unknown id: {rec_id}>")
            rank = rec.get("rank")

            counters_vals = rec.get("counters", [])
            fcounters_vals = rec.get("fcounters", [])

            counters_dict = OrderedDict()
            for i, val in enumerate(counters_vals):
                name = counter_names[i] if i < len(counter_names) else f"counter_{i}"
                if include_zero or val not in (0, -1):
                    counters_dict[name] = val

            fcounters_dict = OrderedDict()
            for i, val in enumerate(fcounters_vals):
                name = fcounter_names[i] if i < len(fcounter_names) else f"fcounter_{i}"
                if include_zero or val != 0.0:
                    fcounters_dict[name] = val

            module_result.append(
                {
                    "file": filename,
                    "id": rec_id,
                    "rank": rank,
                    "counters": counters_dict,
                    "fcounters": fcounters_dict,
                }
            )

        result[module] = module_result

    return result


def print_readable(result, exe=None, jobid=None):
    header = "DARSHAN READABLE REPORT"
    print("=" * 80)
    print(header)
    if exe:
        print(f"Executable: {exe}")
    if jobid is not None:
        print(f"Job ID: {jobid}")
    print("=" * 80)

    for module, recs in result.items():
        print(f"\n--- Module: {module} ---")

        if isinstance(recs, dict) and "info" in recs:
            print(f"  {recs['info']}")
            continue

        if not recs:
            print("  (no records)")
            continue

        for rec in recs:
            print(f"\n  File : {rec['file']}")
            print(f"  Rank : {rec['rank']}")

            if rec["counters"]:
                print("  Counters:")
                for k, v in rec["counters"].items():
                    print(f"    {k:35s} = {v}")
            else:
                print("  Counters: (all zero)")

            if rec["fcounters"]:
                print("  Timers/Fcounters:")
                for k, v in rec["fcounters"].items():
                    print(f"    {k:35s} = {v}")
            else:
                print("  Timers/Fcounters: (all zero)")


def main():
    parser = argparse.ArgumentParser(
        description="Generates a human-readable report from a Darshan JSON file, associating files with their I/O operations."
    )
    parser.add_argument("file", help="Path to the input .darshan.json file")
    parser.add_argument(
        "--json", metavar="OUTPUT", help="Optional path to also save the result in JSON format"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include zero-valued counters/timers (by default they are omitted for readability)",
    )
    args = parser.parse_args()

    data = load_darshan_json(args.file)
    result = build_readable_records(data, include_zero=args.all)

    exe = data.get("metadata", {}).get("exe")
    jobid = data.get("metadata", {}).get("job", {}).get("jobid")

    print_readable(result, exe=exe, jobid=jobid)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResult also saved in JSON format: {args.json}")


if __name__ == "__main__":
    main()