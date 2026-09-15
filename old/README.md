# Launcher generator

Reads a YAML config and writes one SLURM plus Darshan launcher script
(`.sh`) for one of the five training scripts (`hdf5_tr.py`, `bin_tr.py`,
`csv_tr.py`, `npz_tr.py`, `root_tr.py`). This is the first, minimal piece
of what is meant to grow into a full campaign CLI later.

## Requirements

`pyyaml`

## Usage

```bash
python3 generate_launcher.py --config configs/hdf5_example.yaml --output h5/hdf5_compute.sh
```

This only writes the `.sh` file, it does not submit it. Submitting is
still a manual `sbatch path/to/generated.sh`.

## YAML fields

| Field | Meaning | Applies to |
|---|---|---|
| `format` | `hdf5`, `csv`, `npz`, `bin`, or `root` | all |
| `strategy` | `eager`, `lazy`, `stream`, or `burst` | all |
| `model` | `M1`, `M2`, or `M3` | all |
| `batch_size`, `epochs`, `log_every` | ordinary training loop settings | all |
| `workers` | list of worker counts to sweep inside the one generated job | all (see note below for root) |
| `cpus_per_task` | Slurm cores reserved per task | all |
| `dataset_size` | selects `converted_<size>` or the `roots` file naming | all |
| `persistent_workers` | boolean | all |
| `prefetch_factor` | integer, only meaningful for the `burst` strategy | all |
| `block_rows` | integer or omitted (auto), only meaningful for `stream`/`burst` | hdf5, csv, npz, bin |
| `target_col` | label column name | csv, root |
| `tree_name` | TTree/RNTuple object name | root |
| `root_threads` | passed to `--root-threads`, left at 0 unless you have a reason to change it | root |
| `slurm.nodes`, `slurm.time`, `slurm.qos`, `slurm.account`, `slurm.constraint` | `#SBATCH` directives | all |
| `paths.venv` | path to the Python venv to activate | all |

Any field left out falls back to the same default the five hand built
launcher scripts already used.

## What is fixed, not in the YAML

Copied as-is from the launchers already confirmed working on Perlmutter,
not something this generator decides per run: the Athena/ATLAS
environment block, the Darshan environment variables, the per run log and Darshan
directory naming, and the overall loop structure.

## What the generator handles per format automatically

- Where the dataset lives: a single file under `converted_<size>/` for
  hdf5, csv, npz; a directory of shards under `converted_<size>/` for
  bin; the original file under `roots/` for root, the one format whose
  input the converter never writes into its own output folder.
- Which extra flags a format needs (`--target-col`, `--tree-name`).
- What the sanity check after activating the venv imports, matching
  what that format's training script actually depends on.
- root's one structural exception: `--num-workers` only applies to its
  `lazy` strategy, so for `eager`/`stream`/`burst` the generated script
  runs the loop once (labeled `workers_NA`) instead of sweeping a value
  that `root_tr.py` would reject.

## Current scope

- One YAML produces one `.sh`: one format, one strategy, one worker
  sweep list. It does not yet generate a whole campaign matrix (many
  strategies, many formats, many dataset sizes at once), that is a
  planned next step once this piece is trusted.