#!/usr/bin/env python3
"""
Inspect ROOT files containing TTree or RNTuple objects.

For every object stored in a ROOT file, this prints its class name, entry
count, column names and types, and a small preview of the first rows as
plain NumPy values. It does not modify the file in any way.

RDataFrame reads both TTree and RNTuple objects through the same
constructor in recent ROOT versions, so no branching on class name is
needed to read the data; the class name is only used for the printed
label.
"""
import argparse
import ROOT


def inspect_with_rdataframe(path, object_name, max_columns, max_rows):
    """
    Open object_name inside path with RDataFrame and print a summary:
    entry count, a column list truncated to max_columns entries, and up
    to max_rows sample rows read through AsNumpy.
    """
    rdf = ROOT.RDataFrame(object_name, path)

    n_entries = rdf.Count().GetValue()
    print(f"Entries : {n_entries}")

    columns = [str(c) for c in rdf.GetColumnNames()]

    print(f"Columns     : {len(columns)}")

    for c in columns[:max_columns]:
        try:
            ctype = rdf.GetColumnType(c)
        except Exception:
            # A handful of column types do not resolve cleanly through
            # GetColumnType, report as unknown instead of aborting.
            ctype = "unknown"
        print(f"  - {c}: {ctype}")

    if len(columns) > max_columns:
        print(f"  ... ({len(columns) - max_columns} more columns)")

    if not columns:
        return

    print("\n[First rows as NumPy]")
    selected = columns[:max_columns]

    # Range(max_rows) limits how many entries RDataFrame actually reads,
    # so this preview stays cheap even on very large files.
    data = rdf.Range(max_rows).AsNumpy(selected)

    actual_rows = min(max_rows, len(next(iter(data.values()))))

    for i in range(actual_rows):
        print(f"\nRow {i}:")
        for c in selected:
            print(f"  {c} = {data[c][i]}")


def inspect_file(path, max_columns, max_rows):
    """
    Open a ROOT file and inspect every top level object it contains.
    Most files hold exactly one TTree or RNTuple, but this loops over
    every key in case there is more than one.
    """
    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open file: {path}")

    print("\n[Objects in file]")
    f.ls()

    keys = f.GetListOfKeys()
    if keys.GetEntries() == 0:
        print("No objects found.")
        f.Close()
        return

    for key in keys:
        name = key.GetName()
        class_name = key.GetClassName()

        print("\n" + "=" * 40)
        print(f"Object name : {name}")
        print(f"Class       : {class_name}")

        try:
            inspect_with_rdataframe(path, name, max_columns, max_rows)
        except Exception as e:
            # Not every key is a tree like object, it could be a
            # histogram or a directory. Report and continue with the
            # next key instead of aborting the whole file.
            print("\nCould not inspect with RDataFrame.")
            print(f"Reason: {e}")

    f.Close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ROOT files containing TTree or RNTuple objects."
    )

    parser.add_argument("files", nargs="+", help="ROOT files to inspect.")
    parser.add_argument("--max-columns", type=int, default=10, help="Maximum number of columns to print.")
    parser.add_argument("--max-rows", type=int, default=5, help="Maximum number of rows to print.")

    args = parser.parse_args()

    for path in args.files:
        inspect_file(path, args.max_columns, args.max_rows)


if __name__ == "__main__":
    main()
