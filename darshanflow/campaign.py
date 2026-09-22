"""Campaign initialization and detection.

A campaign is a plain filesystem workspace, identified solely by the presence
of campaign.yaml and .darshanflow/manifest.json. There is no global registry.
"""

import json
from datetime import datetime
from pathlib import Path

import yaml

from darshanflow import DarshanFlowError, __version__

# Version of the manifest format; independent of the campaign.yaml schema.
MANIFEST_SCHEMA_VERSION = "1"
CONFIG_FILE = "campaign.yaml"
STATE_DIR = ".darshanflow"
MANIFEST_FILE = f"{STATE_DIR}/manifest.json"
SUBDIRS = ("data", "scripts", "launchers", "runs")

# Human-facing starter configuration (schema v1), kept as text so that its
# comments and ordering reach the user. Consistent with schema1.yaml; the
# only substitution is {name}.
CONFIG_TEMPLATE = r"""schema_version: 1

campaign:
  name: {name}


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

workload:
  type: python
  script: scripts/train.py


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset:
  format: hdf5
  path: data/dataset.h5


# ---------------------------------------------------------------------------
# Fixed workload arguments
#
# This section is intentionally free-form.
#
# Every key is converted using:
#
#   key_name -> KEY_NAME -> --key-name
#
# Example:
#
#   batch_size: 64
#
# becomes conceptually:
#
#   BATCH_SIZE=64
#   --batch-size "$BATCH_SIZE"
#
# Supported values in schema v1:
#   string
#   integer
#   float
#   boolean
#
# true  -> the CLI flag is present
# false -> the CLI flag is omitted
#
# data_path and num_workers are reserved by DarshanFlow and must not
# appear here.
# ---------------------------------------------------------------------------

experiment:
  strategy: eager
  model: M2
  batch_size: 64
  epochs: 1
  log_every: 500
  persistent_workers: true


# ---------------------------------------------------------------------------
# Sweep dimensions
#
# Schema v1 supports only num_workers.
# If num_workers is absent, the workload is executed once without passing
# --num-workers.
# ---------------------------------------------------------------------------

sweep:
  num_workers:
    - 1
    - 2
    - 4
    - 8


# ---------------------------------------------------------------------------
# Environment
#
# setup entries are intentionally raw shell commands. They are an escape
# hatch for machine/site-specific setup that DarshanFlow cannot model yet.
#
# Target-specific setup commands run first, then the virtual environment
# is activated, then variables are exported.
# ---------------------------------------------------------------------------

environment:

  # Optional virtual environment.
  # If null or omitted, no virtual environment is activated.
  # Example: venv: /path/to/venv
  venv: null

  # Optional environment variables exported before running the workload.
  # If empty or omitted, no additional variables are exported.
  # Example:
  #  OMP_NUM_THREADS: 4
  #  MKL_NUM_THREADS: 4
  #  OPENBLAS_NUM_THREADS: 4
  #  NUMEXPR_NUM_THREADS: 4
  variables: {}

  # Optional target-specific shell setup.
  # Commands are inserted verbatim into the generated launcher.
  # If empty or omitted, no setup commands are added.
  # Example:
  #    - export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
  #    - source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
  #    - lsetup prmon
  #    - asetup Athena,main--dev3LCG,latest
  setup:
    local: []
    slurm: []

  # Optional Python modules imported before execution as a sanity check.
  # These do not affect the workload itself.
  # They are only used to verify which Python packages and installations
  # are actually visible in the generated execution environment.
  # Example:
  #  - torch
  #  - h5py
  sanity_imports: []


# ---------------------------------------------------------------------------
# Darshan instrumentation
# ---------------------------------------------------------------------------

darshan:
  enabled: true

  library: /path/to/libdarshan.so

  nonmpi: true
  dxt: true

  config:
    mode: generated
    path: darshan_env.conf

    max_records:
      POSIX: 5000
      DXT_POSIX: 325520

    name_exclude:
      - '\.pyc$'
      - '^/cvmfs'
      - '^/lib64'
      - '^/lib'
      - '^/gpfs/'
      - '^/work2'
      - '^/tmp'
      - '^/dev'
      - '^/scratch1'

    name_include: []

    modmem: 100

    app_exclude:
      - git
      - ls
      - sh
      - hostname
      - sed
      - g++
      - date
      - cc1plus
      - cat
      - which
      - tar
      - ld
      - prmon
      - uname
      - ps
      - rm
      - tee
      - srun

    app_include:
      - python

    rank_exclude: []
    rank_include: []

    dxt_small_io_trigger: 0.01

    dump_config: true


# ---------------------------------------------------------------------------
# Execution backends
# ---------------------------------------------------------------------------

execution:

  local:
    enabled: true

  slurm:
    enabled: true

    nodes: 1
    cpus_per_task: 32
    time: "02:00:00"

    constraint: cpu
    qos: regular
    account: m2845


# ---------------------------------------------------------------------------
# Post-experiment analysis
# ---------------------------------------------------------------------------

analysis:

  metrics:

    io:
      enabled: true

    access:
      enabled: true

    metadata:
      enabled: true

    bandwidth:
      enabled: true

    balance:
      enabled: true

  graphs:

    compact:
      enabled: true

    individual:
      enabled: false
"""


def init_campaign(path: str | Path) -> Path:
    """Create a new campaign workspace at `path` and return its absolute path."""
    root = Path(path).resolve()
    if root.exists() and not root.is_dir():
        raise DarshanFlowError(f"not a directory: {root}")
    if (root / CONFIG_FILE).exists() or (root / STATE_DIR).exists():
        raise DarshanFlowError(f"a DarshanFlow campaign already exists at {root}")

    print(f"Initializing campaign: {root.name}")
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / STATE_DIR).mkdir()
    # str.format() is unusable: the template contains literal braces.
    config = CONFIG_TEMPLATE.replace("{name}", _yaml_scalar(root.name), 1)
    (root / CONFIG_FILE).write_text(config)

    manifest = {
        "campaign": root.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "darshanflow_version": __version__,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Campaign created at: {root}")
    return root


def require_campaign(path: str | Path = ".") -> Path:
    """Return the absolute campaign root at `path`, failing if it is not a campaign."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise DarshanFlowError(f"campaign directory not found: {root}")
    for required in (CONFIG_FILE, MANIFEST_FILE):
        if not (root / required).is_file():
            raise DarshanFlowError(f"not a DarshanFlow campaign (missing {required}): {root}")
    return root


def _yaml_scalar(text: str) -> str:
    """Return `text` as a YAML string scalar, quoted only if plain style would change it.

    Directory names such as 2026, yes or null would otherwise be read back as
    a number, boolean or null instead of a campaign name.
    """
    try:
        if yaml.safe_load(f"key: {text}") == {"key": text}:
            return text
    except yaml.YAMLError:
        pass
    return json.dumps(text, ensure_ascii=False)
