#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --account=m2845

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
# lsetup prmon
asetup Athena,main--dev3LCG,latest

source /pscratch/sd/s/satt/sprints/myvenv/bin/activate

echo "python3 resolved to: $(which python3)"
python3 -c "import torch, h5py; print('torch:', torch.__file__); print('h5py:', h5py.__file__);"

# RUN_ID
WORKFLOW=hdf5
RUN_ID="${WORKFLOW}_$(date +"%Y%m%d_%H%M%S")"

# Configuration
# BASH_SOURCE would resolve to the spool copy Slurm makes of this script,
# not the real folder in /pscratch, so SLURM_SUBMIT_DIR is used instead.
localdir="$SLURM_SUBMIT_DIR"
DATA_DIR=$localdir/../../data/converted_1M
DATA_FILE=mc_normalized_rntuple_1M.h5
DATA_PATH=$DATA_DIR/$DATA_FILE
SCRIPT=$localdir/hdf5_tr.py
OUTPUT_DIR=$localdir/runs/$RUN_ID

echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Script: $SCRIPT"

WORKERS=(2)

CPUS_PER_TASK=32 # number of workers

STRATEGY=eager # eager | lazy | stream | burst
MODEL=M2       # M1 | M2 | M3
BATCHSIZE=64
EPOCHS=1
LOG_EVERY=5000

PERSISTENT_WORKERS_FLAG=""
PREFETCH_FACTOR_ARGS=()
BLOCK_ROWS_ARGS=()

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

mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/darshan_logs

echo "Starting Test..."
echo "Data Path: $DATA_PATH"

SWEEP_VALUES=("${WORKERS[@]}")
for w in "${SWEEP_VALUES[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/logs/log.strategy_${STRATEGY}_workers_${w}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}.txt"
    export DARSHAN_LOGDIR="${OUTPUT_DIR}/darshan_logs/strategy_${STRATEGY}_workers_${w}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}"
    export DARSHAN_LOGPATH="${DARSHAN_LOGDIR}"

    echo "----------------------------------------------------------------"
    echo "Running with Strategy: $STRATEGY | Workers: $w"
    echo "Log: $LOG_FILE"

    export DARSHAN_DUMP_CONFIG=0
    mkdir -p $DARSHAN_LOGDIR
    rm -rf $DARSHAN_LOGDIR/*

    NUM_WORKERS_ARGS=(--num-workers $w)

    srun --cpu-bind=cores --ntasks=1 --cpus-per-task=$CPUS_PER_TASK python3 $SCRIPT \
        --data-path $DATA_PATH \
        --strategy $STRATEGY \
        --model $MODEL \
        --batch-size $BATCHSIZE \
        --epochs $EPOCHS \
        --log-every $LOG_EVERY \
        "${NUM_WORKERS_ARGS[@]}" \
        2>&1 | tee $LOG_FILE
done

deactivate
echo "Test Complete."
