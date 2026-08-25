# Extending ROOT datasets


## Description
1. `extend.py`: scale up RNTuple to N events, repeating events in order until it reaches `--target-events` rows.

 ```bash
   python3 extend.py mc_normalized_rntuple.root mc_normalized_rntuple_10M.root \
       --object-name tree --target-events 10000000 --block-size 100000
   ```

2. `inspect_root.py`: prints entry count, column names/types, and a few sample rows for every TTree/RNTuple object in a ROOT file.

   ```bash
   python3 inspect_root.py mc_normalized_rntuple_10M.root --max-columns 20 --max-rows 3
   ```

3. `dataset_converter.py`: Reads a ROOT file once and
   can write CSV, HDF5, NPZ and raw binary shards. All four converted formats describe the exact same rows, same columns, `X` as `float32`, `y` as
   `int64`.

   ```bash
   # everything in one call, reading the ROOT file exactly once
   python3 dataset_converter.py root-to-all mc_normalized_rntuple_10M.root \
       --tree-name tree --target-col Label \
       --output-dir data/converted_10M --num-shards 1
   ```

   Individual steps are also available: `root-to-csv`, `root-to-hdf5`,
   `root-to-npz`, `root-to-bin`, `root-to-all`, and the CSV-sourced
   equivalents `csv-to-hdf5`, `csv-to-npz`, `csv-to-bin`, `csv-to-all` (for
   when you already have a CSV and don't want to re-read the ROOT file).
   See `--help` on each subcommand for the full flag list.


## Data folder suggested structure

```
data/
├── mc_normalized_rntuple_{1,5,10,50}M.root   # extend.py output, one per scale
└── converted_{1,5,10,50}M/                    # dataset_converter.py output, one dir per scale
    ├── mc_normalized_rntuple_{N}M.csv
    ├── mc_normalized_rntuple_{N}M.h5
    ├── mc_normalized_rntuple_{N}M.h5.metadata.json
    ├── mc_normalized_rntuple_{N}M.npz
    ├── mc_normalized_rntuple_{N}M.npz.metadata.json
    └── mc_normalized_rntuple_{N}M_bin/
        ├── shard_000.bin, shard_001.bin, ...   # one file if --num-shards 1
        └── metadata.json
```

## Design notes relevant to the Dataset/DataLoader implementations

- **Metadata JSON**: every format gets a metadata sidecar
  (`<file>.metadata.json` for HDF5/NPZ, `metadata.json` inside the shard
  directory for BIN) with `num_samples`, `num_features`, dtypes, and
  feature names. This means `_compute_length()` in the Lazy dataset
  classes can read a small JSON instead of opening a multi-GB file just to
  ask its shape.
- **CSV**: has no equivalent cheap shape lookup, row count requires
  either scanning the file once or trusting an external metadata source.