# Darshan log validation and aggregation

## Description

The scripts in this directory were used to inspect Darshan logs and aggregate I/O counters during the experimental campaigns.

### `dlogcheck.sh`

Checks a single Darshan log for incomplete module data.

The script runs `darshan-parser --show-incomplete`, saves the parser output to `<logfile>.txt`, and reports whether the log contains the `module contains incomplete data` warning.

```bash
./dlogcheck.sh run.darshan
```

### `fdlogcheck.sh`

Runs the same incomplete-data check over multiple Darshan logs.

```bash
./fdlogcheck.sh runs/*.darshan
```

A `.txt` parser output is created next to each input log.

### `ops_per_file.py`

Reads a Darshan JSON file and associates the records in each module with their corresponding file names from `name_records`.

For each record, the output includes the file path, rank, record ID, counters, and floating-point counters. Zero-valued counters are omitted by default.

```bash
python3 ops_per_file.py run.darshan.json
```

Use `--json` to save the processed records:

```bash
python3 ops_per_file.py run.darshan.json \
    --json run_ops.json
```

Use `--all` to include zero-valued counters and timers:

```bash
python3 ops_per_file.py run.darshan.json --all
```

The expected input JSON contains the Darshan `name_records`, `counters`, and `records` sections.

### `aggregate.py`

Aggregates counters from a collection of `*_ops.json` files.

Records are selected by matching a string against their `file` field. By default the match is a substring; `--exact` restricts it to an exact path or basename match.

For example, to aggregate records for HDF5 files in the current directory:

```bash
python3 aggregate.py .h5
```

To read the `*_ops.json` files from another directory:

```bash
python3 aggregate.py .h5 --dir /path/to/ops_json
```

To save the aggregate result as JSON:

```bash
python3 aggregate.py .h5 \
    --dir /path/to/ops_json \
    --json accumulated_h5.json
```

An exact filename can also be selected:

```bash
python3 aggregate.py mc_normalized_rntuple_10M.h5 --exact
```

The output keeps the accumulated counters together with the matched paths, ranks, source `*_ops.json` files, and source record IDs.

`fcounters` are summed in the generated aggregate. Sums of `*_TIME` values represent accumulated durations, but sums of absolute `*_TIMESTAMP` values do not have a useful physical interpretation.

## Typical workflow

The scripts can be used in this order:

```text
.darshan log
    |
    |  dlogcheck.sh / fdlogcheck.sh
    v
log completeness check

Darshan JSON
    |
    |  ops_per_file.py
    v
*_ops.json
    |
    |  aggregate.py
    v
aggregated counters for selected files
```

The conversion from a raw `.darshan` log to the Darshan JSON consumed by `ops_per_file.py` is handled separately.