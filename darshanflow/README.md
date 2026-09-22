# darshanflow package

This package holds the lifecycle stages behind `cli.py`. Each CLI command maps
to one module, and `config.py` is the only module that interprets
`campaign.yaml`.

```
cli.py ─┬─ init    → campaign.py
        ├─ build   → builder.py  ── uses config.py
        ├─ run     → runner.py   ── uses builder.bundle_path()
        └─ analyze → analysis.py ── uses config.py, runs post/* as subprocesses
```

`__init__.py` defines `__version__` and `DarshanFlowError`. The CLI prints that
error as `error: ...` without a traceback.

## campaign.py: `init` and campaign detection

- `init_campaign(path)` creates `data/`, `scripts/`, `launchers/`, `runs/`,
  `.darshanflow/manifest.json`, and a starter `campaign.yaml` that matches
  `schema1.yaml`, with the campaign name filled in. It refuses to overwrite an
  existing campaign.
- `require_campaign(path)` treats a directory as a campaign when it has both
  `campaign.yaml` and the manifest. Every other command calls it.

## config.py: schema-v1 validation

`load_campaign_config(root)` / `load_config_file(path, root)` return a
normalized nested dict. Validation is strict:

- Unknown keys, wrong types and duplicate YAML keys are errors. Error messages
  use dotted paths (`execution.slurm.nodes`) and suggest a key when the name
  is close to a valid one.
- **Structure has defaults, behavior does not.** Every `enabled` flag and every
  analysis toggle must be written explicitly.
- Optional sections (`experiment`, `sweep`, `environment`) that are left out
  or empty become `None`.
- A feature's settings are required only while it is enabled (`darshan.*`,
  `execution.slurm.*`). While it is disabled, they are type-checked and then
  dropped.
- Relative paths resolve against the campaign root, not the CWD. `$VARS` are
  rejected, and paths don't need to exist yet.
- `experiment` is the only open section. Each key maps to
  `batch_size → BATCH_SIZE → --batch-size`. `true` passes the bare flag,
  `false` omits it. `data_path` and `num_workers` are reserved.
- YAML base-60 parsing is disabled, so `12:15` and `02:00:00` stay strings.

## builder.py: `build`

This module turns a valid config into a **build bundle**
`launchers/build-<ID>/`. The ID is `sha256(campaign.yaml bytes + version +
BUILD_FORMAT_VERSION)[:12]`.

- It renders everything in memory first and only then writes files
  atomically. A failed build writes nothing.
- Only targets that are enabled in `execution` are built. Asking for a
  disabled target is an error.
- It rejects experiment keys or environment variables that would overwrite
  shell variables the launcher uses (`PATH`, `SLURM_*`, `RUN_ID`, ...).
- In `generated` mode it writes `darshan_env.conf` from `darshan.config`. In
  `external` mode it points the launcher at your file.

Each generated launcher (`local.sh`, `slurm.sh`) does the following, in order:

1. Finds its bundle and the campaign root (`$localdir`) from its own path, so
   it works from any CWD. `slurm.sh` falls back to the build-time path when
   sbatch runs a spooled copy.
2. Runs `environment.setup.<target>` verbatim, activates the venv, exports
   variables and runs the sanity imports.
3. Sets `SCRIPT`, `DATA_PATH` and the experiment variables, and builds
   `WORKLOAD_ARGS`.
4. Checks that the Darshan library exists, then creates
   `runs/<format>_<timestamp>/` and copies `config.yaml`, `launcher.sh` and the
   Darshan config into it.
5. For each case, sets `DARSHAN_LOGDIR`, sets `LD_PRELOAD` only around the
   workload, and runs `python3` (wrapped in `srun --ntasks=1` on Slurm). The
   output is teed to `logs/<case>.txt`.

Paths inside the campaign are written relative to `$localdir`, so the
campaign can be moved. External paths stay absolute.

**Intentional gaps:** `rank_include` / `rank_exclude` are not written, and only
a commented example is kept. `campaign.name` is not used, and only
`workload.type: python` is supported.

## runner.py: `run`

This is a thin dispatcher and does no validation.

- It hashes the current `campaign.yaml` bytes and finds the matching bundle.
  It then checks that the bundle snapshot is byte-identical and that the
  launcher for the target exists.
- It gives separate errors for "never built / config changed" and "target not
  built".
- `local` runs `bash <bundle>/local.sh` synchronously. `slurm` runs
  `sbatch <bundle>/slurm.sh`, which only submits the job.
- `--dry-run` runs every check and prints the command without executing it.
- It never writes anything. The launcher owns the run directory.

## analysis.py: `analyze`

This module orchestrates the pipeline for one run (`--run <ID>` or `latest`,
where latest is taken from the ID timestamp).

1. It loads the run's `config.yaml` snapshot and fails if Darshan was
   disabled in it.
2. It selects metrics and graph modes. By default these are the ones the
   snapshot enables. `--metrics` / `--graphs` may only pick a subset.
3. It derives the expected cases from the snapshot. Each case must contain
   at least one `*.darshan` file; otherwise the run is reported as incomplete.
4. For each case, it recreates `analysis/<case>/` and runs these steps:
   - `checks/`: `darshan-parser --show-incomplete`. Any incomplete log fails
     the case.
   - `json/`: `python -m darshan to_json`.
   - `simplified.json`: `post.simplify_logs`, keeping only records whose path
     matches the workload script or the dataset.
   - `<metric>/`: the selected `post.*` modules.
5. Cases are independent. A failed case doesn't stop the others, but the
   command exits 1 at the end.

Metric names map to modules as follows: `io` → io_intensity, `access` →
access_pattern, `metadata` → metadata_pressure, `bandwidth` →
effective_read, `balance` → worker_balance. `io` draws graphs only in
`individual` mode.
