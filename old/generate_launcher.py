#!/usr/bin/env python3
"""
Generates a SLURM launcher script (.sh) for one of the five training
scripts (hdf5_tr.py, bin_tr.py, csv_tr.py, npz_tr.py, root_tr.py) from a
YAML config, so that a launcher no longer has to be hand written and
copied between formats.

This is the first, minimal piece of what is meant to grow into a full
campaign CLI. Today it covers exactly one job per YAML file (one format,
one strategy, one worker sweep list), matching the granularity of the
five launcher scripts already hand built and confirmed working on
Perlmutter. 

Everything this script writes into the boilerplate parts of the launcher
(the Athena/ATLAS environment block, the Darshan environment variables,
the per run log/darshan directory handling, the venv activation) is
copied from the five launchers already confirmed to run correctly, not
redesigned here. The only thing this script decides dynamically is the
handful of lines that differ between formats, the same differences
already mapped by hand across those five files: where the dataset lives,
which extra flags a given script needs, what the sanity check should
import, and the one structural exception root_tr.py has (num_workers
only applies to its lazy strategy).

Usage:
  python3 generate_launcher.py --config configs/hdf5_example.yaml --output hdf5/hdf5_compute.sh

"""

import argparse
import os

import yaml


# ---------------------------------------------------------------------------
# What is fixed, copied verbatim from the five confirmed launchers
# ---------------------------------------------------------------------------

ATHENA_BLOCK = """\
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${{ATLAS_LOCAL_ROOT_BASE}}/user/atlasLocalSetup.sh
lsetup prmon
asetup Athena,main--dev3LCG,latest

source {venv}/bin/activate
"""

DARSHAN_BLOCK = """\
# Setup Darshan environment. DXT is enabled both through darshan_env.conf's
# own MOD_ENABLE line and through this env var, redundant on purpose, see
# the project README for why neither alone was fully trusted. APP_EXCLUDE
# in darshan_env.conf was updated to also skip rm/srun/tee, since exporting
# LD_PRELOAD before the loop (needed so srun inherits it) means every
# process in this shell after this point gets instrumented, not just
# python3, unless explicitly excluded there.
export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_CONFIG_PATH=$localdir/darshan_env.conf
export LD_PRELOAD=/global/cfs/cdirs/m2845/darshan/darshan-main/lib/libdarshan.so
export DXT_ENABLE_IO_TRACE=1
"""


# ---------------------------------------------------------------------------
# What differs per format, mapped from the five hand built launchers
# ---------------------------------------------------------------------------

FORMAT_SPECS = {
    "hdf5": {
        "script": "hdf5_tr.py",
        "ext": "h5",
        "data_kind": "converted_file",
        "sanity_imports": ["torch", "h5py"],
        "extra_cli": [],
        "supports_block_rows": True,
        "supports_root_threads": False,
        "worker_applies_to_all_strategies": True,
    },
    "csv": {
        "script": "csv_tr.py",
        "ext": "csv",
        "data_kind": "converted_file",
        "sanity_imports": ["torch", "pandas"],
        "extra_cli": [("target_col", "--target-col")],
        "supports_block_rows": True,
        "supports_root_threads": False,
        "worker_applies_to_all_strategies": True,
    },
    "npz": {
        "script": "npz_tr.py",
        "ext": "npz",
        "data_kind": "converted_file",
        "sanity_imports": ["torch", "numpy"],
        "extra_cli": [],
        "supports_block_rows": True,
        "supports_root_threads": False,
        "worker_applies_to_all_strategies": True,
    },
    "bin": {
        "script": "bin_tr.py",
        "ext": None,
        "data_kind": "converted_dir",
        "sanity_imports": ["torch", "numpy"],
        "extra_cli": [],
        "supports_block_rows": True,
        "supports_root_threads": False,
        "worker_applies_to_all_strategies": True,
    },
    "root": {
        "script": "root_tr.py",
        "ext": "root",
        "data_kind": "roots_file",
        "sanity_imports": ["torch", "ROOT"],
        "extra_cli": [("target_col", "--target-col"), ("tree_name", "--tree-name")],
        "supports_block_rows": False,
        "supports_root_threads": True,
        "worker_applies_to_all_strategies": False,
    },
}


def build_sanity_check(imports):
    """
    Mirrors each launcher's own sanity check: confirm which python3 and
    which of the format's key packages actually get picked up after
    activating the venv on top of the Athena environment, since Athena
    provides its own versions of most of these and could shadow the
    venv's silently otherwise.
    """
    import_line = ", ".join(imports)
    print_parts = " ".join(
        "print('{name}:', {name}.__file__);".format(name=name) for name in imports
    )
    return (
        'echo "python3 resolved to: $(which python3)"\n'
        'python3 -c "import {imports}; {prints}"\n'
    ).format(imports=import_line, prints=print_parts)


def build_data_path_block(fmt, spec, dataset_size):
    """
    Returns the DATA_DIR/DATA_PATH assignment lines for this format,
    mirroring the three distinct layouts already confirmed by hand:
    a single converted file (hdf5, csv, npz), a converted directory of
    shards (bin), or the original source file under data/roots (root,
    the only one of the five whose input was never written into a
    converted_<size> folder by the dataset converter at all).
    """
    if spec["data_kind"] == "roots_file":
        lines = [
            "# Unlike the other four formats, the ROOT file is not under",
            "# converted_{size}, the converter reads from this roots/ folder,".format(size=dataset_size),
            "# it never writes a .root file into its own output directory.",
            "DATA_DIR=$localdir/../../data/roots",
            "DATA_FILE=mc_normalized_rntuple_{size}.{ext}".format(size=dataset_size, ext=spec["ext"]),
            "DATA_PATH=$DATA_DIR/$DATA_FILE",
        ]
    elif spec["data_kind"] == "converted_dir":
        lines = [
            "# --data-path for bin_tr.py is a directory (shard_*.bin plus",
            "# metadata.json), not a single file, unlike the other formats.",
            "DATA_DIR=$localdir/../../data/converted_{size}".format(size=dataset_size),
            "BIN_SUBDIR=mc_normalized_rntuple_{size}_bin".format(size=dataset_size),
            "DATA_PATH=$DATA_DIR/$BIN_SUBDIR",
        ]
    else:
        lines = [
            "DATA_DIR=$localdir/../../data/converted_{size}".format(size=dataset_size),
            "DATA_FILE=mc_normalized_rntuple_{size}.{ext}".format(size=dataset_size, ext=spec["ext"]),
            "DATA_PATH=$DATA_DIR/$DATA_FILE",
        ]
    return "\n".join(lines) + "\n"


def build_extra_cli_lines(spec, config):
    """
    Variable assignments plus their corresponding CLI flags, for the
    format specific arguments mapped in FORMAT_SPECS (--target-col for
    csv and root, --tree-name for root only, and so on). Values are read
    from the YAML, falling back to the same defaults the hand built
    launchers used. Returns "" for both parts when a format has no extra
    arguments (hdf5, npz, bin), so the caller can skip that block of the
    template entirely rather than emit an empty continuation line.
    """
    defaults = {"target_col": "Label", "tree_name": "tree"}
    assign_lines = []
    cli_lines = []
    for yaml_key, flag in spec["extra_cli"]:
        value = config.get(yaml_key, defaults.get(yaml_key))
        var_name = yaml_key.upper()
        assign_lines.append("{var}={value}".format(var=var_name, value=value))
        cli_lines.append("        {flag} ${var} \\".format(flag=flag, var=var_name))
    return "\n".join(assign_lines), "\n".join(cli_lines)


def build_optional_cli_block(config, spec):
    """
    persistent_workers, prefetch_factor and block_rows are each either
    entirely absent from the command line (their argparse default takes
    over) or present as one flag, matching how every hand built launcher
    left them off unless a specific strategy needed them.
    """
    lines = []
    bash_array_lines = []

    if config.get("persistent_workers", False):
        bash_array_lines.append('PERSISTENT_WORKERS_FLAG="--persistent-workers"')
    else:
        bash_array_lines.append('PERSISTENT_WORKERS_FLAG=""')

    prefetch_factor = config.get("prefetch_factor")
    if prefetch_factor is not None:
        bash_array_lines.append("PREFETCH_FACTOR_ARGS=(--prefetch-factor {v})".format(v=prefetch_factor))
    else:
        bash_array_lines.append("PREFETCH_FACTOR_ARGS=()")

    if spec["supports_block_rows"]:
        block_rows = config.get("block_rows")
        if block_rows is not None:
            bash_array_lines.append("BLOCK_ROWS_ARGS=(--block-rows {v})".format(v=block_rows))
        else:
            bash_array_lines.append("BLOCK_ROWS_ARGS=()")

    return "\n".join(bash_array_lines) + "\n"


def render_launcher(config):
    fmt = config["format"]
    if fmt not in FORMAT_SPECS:
        raise ValueError("Unknown format: {}, expected one of {}".format(fmt, list(FORMAT_SPECS)))
    spec = FORMAT_SPECS[fmt]

    strategy = config["strategy"]
    model = config.get("model", "M2")
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 1)
    log_every = config.get("log_every", 200)
    cpus_per_task = config.get("cpus_per_task", 32)
    workers = config.get("workers", [2])
    dataset_size = config.get("dataset_size", "1M")
    venv = config.get("paths", {}).get("venv", "/pscratch/sd/s/satt/sprints/myvenv")

    slurm = config.get("slurm", {})
    nodes = slurm.get("nodes", 1)
    time_limit = slurm.get("time", "00:30:00")
    qos = slurm.get("qos", "debug")
    account = slurm.get("account", "m2845")
    constraint = slurm.get("constraint", "cpu")

    extra_assign, extra_cli = build_extra_cli_lines(spec, config)
    optional_cli_block = build_optional_cli_block(config, spec)

    root_threads_assign = ""
    root_threads_cli = ""
    if spec["supports_root_threads"]:
        root_threads = config.get("root_threads", 0)
        root_threads_assign = (
            "\n# Kept fixed at 0 by default on purpose: EnableImplicitMT's effect on\n"
            "# RDataLoader specifically is unconfirmed, see root_tr.py's own\n"
            "# --root-threads help text and the project README.\n"
            "ROOT_THREADS={v}\n"
        ).format(v=root_threads)
        root_threads_cli = "        --root-threads $ROOT_THREADS \\\n"

    optional_cli_lines = ""
    if config.get("persistent_workers", False):
        optional_cli_lines += "        $PERSISTENT_WORKERS_FLAG \\\n"
    if config.get("prefetch_factor") is not None:
        optional_cli_lines += '        "${PREFETCH_FACTOR_ARGS[@]}" \\\n'
    if spec["supports_block_rows"] and config.get("block_rows") is not None:
        optional_cli_lines += '        "${BLOCK_ROWS_ARGS[@]}" \\\n'

    data_path_block = build_data_path_block(fmt, spec, dataset_size)
    sanity_check = build_sanity_check(spec["sanity_imports"])

    header = """\
#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --time={time_limit}
#SBATCH --constraint={constraint}
#SBATCH --qos={qos}
#SBATCH --account={account}

{athena_block}
{sanity_check}
# RUN_ID
WORKFLOW={fmt}
RUN_ID="${{WORKFLOW}}_$(date +"%Y%m%d_%H%M%S")"

# Configuration
# BASH_SOURCE would resolve to the spool copy Slurm makes of this script,
# not the real folder in /pscratch, so SLURM_SUBMIT_DIR is used instead.
localdir="$SLURM_SUBMIT_DIR"
{data_path_block}SCRIPT=$localdir/{script}
OUTPUT_DIR=$localdir/runs/$RUN_ID

echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Script: $SCRIPT"

WORKERS=({workers})

CPUS_PER_TASK={cpus_per_task}

STRATEGY={strategy} # eager | lazy | stream | burst
MODEL={model}       # M1 | M2 | M3
BATCHSIZE={batch_size}
EPOCHS={epochs}
LOG_EVERY={log_every}
{extra_assign}
{optional_cli_block}{root_threads_assign}
{darshan_block}
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/darshan_logs

echo "Starting Test..."
echo "Data Path: $DATA_PATH"
""".format(
        nodes=nodes,
        time_limit=time_limit,
        constraint=constraint,
        qos=qos,
        account=account,
        athena_block=ATHENA_BLOCK.format(venv=venv),
        sanity_check=sanity_check,
        fmt=fmt,
        data_path_block=data_path_block,
        script=spec["script"],
        workers=" ".join(str(w) for w in workers),
        cpus_per_task=cpus_per_task,
        strategy=strategy,
        model=model,
        batch_size=batch_size,
        epochs=epochs,
        log_every=log_every,
        extra_assign=("\n" + extra_assign if extra_assign else ""),
        optional_cli_block=optional_cli_block,
        root_threads_assign=root_threads_assign,
        darshan_block=DARSHAN_BLOCK,
    )

    if spec["worker_applies_to_all_strategies"]:
        loop_setup = ""
        sweep_values_line = 'SWEEP_VALUES=("${WORKERS[@]}")\n'
        num_workers_block = (
            "    NUM_WORKERS_ARGS=(--num-workers $w)\n"
        )
    else:
        # root_tr.py: num_workers only applies to strategy lazy. Mirrors
        # root_compute.sh's own conditional exactly, rather than pretending
        # a worker sweep applies to eager/stream/burst when it does not.
        loop_setup = (
            'if [ "$STRATEGY" = "lazy" ]; then\n'
            "    SWEEP_VALUES=(\"${WORKERS[@]}\")\n"
            "else\n"
            "    # eager, stream, burst: num_workers/persistent_workers are\n"
            "    # rejected by root_tr.py for these three, so this loop runs\n"
            "    # exactly once, labeled workers_NA instead of iterating a\n"
            "    # value that would not do anything.\n"
            '    SWEEP_VALUES=("NA")\n'
            "fi\n\n"
        )
        sweep_values_line = ""
        num_workers_block = (
            '    NUM_WORKERS_ARGS=()\n'
            '    if [ "$STRATEGY" = "lazy" ]; then\n'
            "        NUM_WORKERS_ARGS=(--num-workers $w)\n"
            "    fi\n"
        )

    loop = """\
{loop_setup}{sweep_values_line}for w in "${{SWEEP_VALUES[@]}}"; do
    LOG_FILE="${{OUTPUT_DIR}}/logs/log.strategy_${{STRATEGY}}_workers_${{w}}_batchsize_${{BATCHSIZE}}_epochs_${{EPOCHS}}.txt"
    export DARSHAN_LOGDIR="${{OUTPUT_DIR}}/darshan_logs/strategy_${{STRATEGY}}_workers_${{w}}_batchsize_${{BATCHSIZE}}_epochs_${{EPOCHS}}"
    export DARSHAN_LOGPATH="${{DARSHAN_LOGDIR}}"

    echo "----------------------------------------------------------------"
    echo "Running with Strategy: $STRATEGY | Workers: $w"
    echo "Log: $LOG_FILE"

    export DARSHAN_DUMP_CONFIG=0
    mkdir -p $DARSHAN_LOGDIR
    rm -rf $DARSHAN_LOGDIR/*

{num_workers_block}
    srun --cpu-bind=cores --ntasks=1 --cpus-per-task=$CPUS_PER_TASK python3 $SCRIPT \\
        --data-path $DATA_PATH \\
{extra_cli_block}        --strategy $STRATEGY \\
        --model $MODEL \\
        --batch-size $BATCHSIZE \\
        --epochs $EPOCHS \\
{root_threads_cli}        --log-every $LOG_EVERY \\
        "${{NUM_WORKERS_ARGS[@]}}" \\
{optional_cli_lines}        2>&1 | tee $LOG_FILE
done

deactivate
echo "Test Complete."
""".format(
        loop_setup=loop_setup,
        sweep_values_line=sweep_values_line,
        num_workers_block=num_workers_block,
        extra_cli_block=(extra_cli + "\n" if extra_cli else ""),
        root_threads_cli=root_threads_cli,
        optional_cli_lines=optional_cli_lines,
    )

    return header + "\n" + loop


def main():
    parser = argparse.ArgumentParser(description="Generate a SLURM+Darshan launcher script from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--output", required=True, help="Path to write the generated .sh file to.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    script_text = render_launcher(config)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(script_text)

    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()