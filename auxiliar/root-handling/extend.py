#!/usr/bin/env python3
"""
Extend a ROOT RNTuple by repeating its events in ordered passes.

Produces a new RNTuple with exactly target_events rows by replaying the
events already present in the input file, in their original order, in as
many full or partial passes as needed. This is not random resampling and
does not generate new or independent values: every value in row i is
identical to the value in row (i + n_original), where n_original is the
number of events in the input file. 
"""
import argparse
import ROOT


# ROOT column types this script knows how to read out of RDataFrame and
# write back into an RNTupleWriter entry. A column whose type is not in
# this map raises immediately rather than being silently skipped or
# miswritten.
SUPPORTED_TYPES = {
    "double": "double",
    "float": "float",
    "std::int32_t": "std::int32_t",
    "int": "int",
    "unsigned int": "unsigned int",
    "std::uint32_t": "std::uint32_t",
    "std::int64_t": "std::int64_t",
    "Long64_t": "Long64_t",
    "bool": "bool",
    "std::string": "std::string",
}


def get_schema(rdf):
    """Return (columns, schema), mapping every column name in rdf to its
    RNTuple field type string, validated against SUPPORTED_TYPES."""
    columns = [str(c) for c in rdf.GetColumnNames()]
    schema = {}

    for col in columns:
        ctype = str(rdf.GetColumnType(col))

        if ctype not in SUPPORTED_TYPES:
            raise TypeError(f"Unsupported column type for column '{col}': {ctype}")

        schema[col] = SUPPORTED_TYPES[ctype]

    return columns, schema


def create_model(columns, schema):
    """Build an RNTupleModel that mirrors columns one to one, in order."""
    model = ROOT.RNTupleModel.Create()

    for col in columns:
        model.MakeField[schema[col]](col)

    return model


def cast_value(value, ctype):
    """Cast a single AsNumpy scalar value to the Python type
    RNTupleWriter expects for ctype, so assigning it to an entry field
    works cleanly."""
    if ctype in {
        "std::int32_t",
        "int",
        "unsigned int",
        "std::uint32_t",
        "std::int64_t",
        "Long64_t",
    }:
        return int(value)

    if ctype in {"double", "float"}:
        return float(value)

    if ctype == "bool":
        return bool(value)

    if ctype == "std::string":
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    raise TypeError(f"Unsupported type during cast: {ctype}")


def write_arrays_to_rntuple(writer, entry, columns, schema, arrays, n_rows):
    """Write n_rows rows from arrays, a dict mapping column name to numpy
    array as returned by AsNumpy, into writer, one Fill per row."""
    for i in range(n_rows):
        for col in columns:
            entry[col] = cast_value(arrays[col][i], schema[col])
        writer.Fill(entry)


def extend_rntuple(input_path, output_path, object_name, target_events, block_size):
    """
    Repeat the events in input_path in order, in as many full or partial
    passes as needed, until output_path contains exactly target_events
    rows.

    block_size only controls how many rows are read into memory per
    AsNumpy call while generating the output; it changes memory use and
    generation speed, not the resulting file's content.
    """
    rdf = ROOT.RDataFrame(object_name, input_path)

    n_original = rdf.Count().GetValue()
    columns, schema = get_schema(rdf)

    if target_events <= 0:
        raise ValueError("target_events must be positive.")

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    print(f"Input           : {input_path}")
    print(f"Output          : {output_path}")
    print(f"Object          : {object_name}")
    print(f"Original events : {n_original}")
    print(f"Target events   : {target_events}")
    print(f"Block size      : {block_size}")
    print(f"Columns         : {len(columns)}")

    if target_events % n_original != 0:
        remainder = target_events % n_original
        print(
            f"Note: target_events is not a multiple of the original event "
            f"count, the final pass will only replay the first {remainder} "
            f"original rows."
        )

    for col in columns:
        print(f"  - {col}: {schema[col]}")

    model = create_model(columns, schema)

    written = 0
    pass_id = 0

    with ROOT.RNTupleWriter.Recreate(model, object_name, output_path) as writer:
        entry = writer.CreateEntry()

        while written < target_events:
            pass_id += 1
            remaining_total = target_events - written
            # Every pass replays the original dataset from its start, or
            # a truncated prefix of it on the final partial pass. This is
            # what creates the repeating period described in the module
            # docstring.
            pass_events = min(n_original, remaining_total)

            print(f"\nPass {pass_id}: writing {pass_events} events")

            start = 0

            while start < pass_events:
                stop = min(start + block_size, pass_events)
                n_rows = stop - start

                print(f"  input rows {start}:{stop} maps to output rows {written}:{written + n_rows}")

                arrays = rdf.Range(start, stop).AsNumpy(columns)

                write_arrays_to_rntuple(
                    writer=writer,
                    entry=entry,
                    columns=columns,
                    schema=schema,
                    arrays=arrays,
                    n_rows=n_rows,
                )

                written += n_rows
                start = stop

    print(f"\nDone. Written events: {written}")


def main():
    parser = argparse.ArgumentParser(
        description="Extend a ROOT RNTuple by repeating its events in ordered passes."
    )

    parser.add_argument("input", help="Input ROOT file containing an RNTuple.")
    parser.add_argument("output", help="Output ROOT file to create.")
    parser.add_argument("--object-name", default="tree", help="Name of the RNTuple object.")
    parser.add_argument(
        "--target-events", type=int, required=True,
        help="Number of events in the extended output dataset.",
    )
    parser.add_argument(
        "--block-size", type=int, default=100000,
        help="Number of input rows read per block.",
    )

    args = parser.parse_args()

    extend_rntuple(
        input_path=args.input,
        output_path=args.output,
        object_name=args.object_name,
        target_events=args.target_events,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
