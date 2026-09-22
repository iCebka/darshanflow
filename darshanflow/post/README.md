# darshanflow.post: post-processing modules

These are the production copies of the project's original manual scripts.
Apart from line endings they are unchanged. Each one is a standalone CLI.
`analysis.py` runs them as subprocesses (`python -m darshanflow.post.<module>`),
and you can also run them by hand on any simplified JSON.

None of these modules reads `.darshan` files directly. The flow is:

```
*.darshan ─(python -m darshan to_json)─> *.darshan.json ─(simplify_logs)─> simplified.json ─> metric modules ─> CSV (+ PNG)
```

## simplify_logs.py

This module normalizes the logs and does no metric computation.

```bash
python -m darshanflow.post.simplify_logs <json_dir> [-i INCLUDE_REGEX] [-e EXCLUDE_REGEX] [-o simplified.json]
```

- It recursively reads every `*.darshan.json` file under `<json_dir>`.
- It keeps the POSIX records whose file name matches `--include` and then
  drops those that match `--exclude`.
- It tags each process log with a role: `parent`, or `worker` for forked
  DataLoader workers. It also parses experiment coordinates, such as
  `--num-workers`, from the command line.
- It writes a single JSON file with named counters that the metric modules
  consume.

## Metric modules

Every metric module takes the same input and options:

```bash
python -m darshanflow.post.<module> simplified.json [-o OUTDIR] [--compact-graphs true|false] [--individual-graphs true|false]
```

`io_intensity` uses a single `--graphs true|false` option instead of the two
graph options.

The modules write `file_metrics.csv` (per run/pid/file), `proc_metrics.csv`
(per process) and `run_metrics.csv` (per run). Graphs are drawn from those
CSVs and require `matplotlib`:

- `graphs/compact/` holds run-level and cross-worker summaries.
- `graphs/individual/` holds one plot per file, process and run.

| Module (`analyze` name)          | Measures                                                         |
|----------------------------------|------------------------------------------------------------------|
| `io_intensity` (`io`)            | Access-size histograms and operation counts                       |
| `access_pattern` (`access`)      | Consecutive / sequential-with-gaps / non-sequential reads, seeks per 1000 reads |
| `metadata_pressure` (`metadata`) | Opens and stats as a share of tracked ops, per 1000 reads, and meta time vs. runtime |
| `effective_read` (`bandwidth`)   | Effective read bandwidth, run throughput and read time vs. runtime |
| `worker_balance` (`balance`)     | Bytes, reads and read-time spread across DataLoader workers (mean, std, min/max, CV). This module writes no `file_metrics.csv`. |

All the modules follow two rules:

- A counter value of `-1` means Darshan couldn't collect it, and it is
  treated as 0 before summing.
- Ratios come from summed absolute counters. Per-process percentages are
  never averaged.

## Limitations

- Only POSIX counters are used. MPI-IO and rank-imbalance counters are ignored.
- `worker_balance` excludes parent processes from its statistics.
- Each invocation covers one case. The modules don't compare cases or runs
  with each other.
- `io_intensity` has no compact graphs.
