"""Build bundles: materialize a validated campaign.yaml into launchers.

build_campaign() validates campaign.yaml through config.py (the only schema
authority), renders every requested artifact in memory, and only then writes
a deterministic build bundle:

    launchers/build-<ID>/
        campaign.yaml        byte-for-byte snapshot of the source
        local.sh, slurm.sh   requested and enabled targets
        darshan_env.conf     generated Darshan config (generated mode only)

<ID> derives from the exact campaign.yaml bytes, the DarshanFlow version and
BUILD_FORMAT_VERSION. Bundles are provenance and are never deleted;
rebuilding the same configuration rewrites identical artifacts in place.

Launchers never depend on the working directory: they locate their bundle
from their own path, and the campaign root (localdir) two levels above it.
Campaign-internal paths are rendered from $localdir, external paths stay
absolute. Nothing is executed and no configured path must exist at build time.
"""

import hashlib
import os
import shlex
from pathlib import Path

from darshanflow import DarshanFlowError, __version__
from darshanflow.campaign import CONFIG_FILE, require_campaign
from darshanflow.config import (
    EXECUTION_TARGETS, experiment_cli_flag, experiment_shell_variable, load_campaign_config,
)

# Bump whenever generated artifacts change for an identical campaign.yaml.
BUILD_FORMAT_VERSION = "1"
BUILD_ID_LENGTH = 12
LAUNCHERS_DIR = "launchers"
# Public alias of the execution targets, used by cli.py and runner.py.
TARGETS = EXECUTION_TARGETS

_SBATCH_OPTIONS = (
    ("nodes", "nodes"), ("cpus_per_task", "cpus-per-task"), ("time", "time"),
    ("constraint", "constraint"), ("qos", "qos"), ("account", "account"),
)
# Entries DarshanFlow itself writes into a bundle or a run directory.
_BUNDLE_ENTRIES = {CONFIG_FILE, *(f"{target}.sh" for target in TARGETS)}
_RUN_ENTRIES = {"config.yaml", "launcher.sh", "logs", "darshan_logs"}

# Shell variables defined by every generated launcher.
_LAUNCHER_VARIABLES = {
    "BUNDLE_DIR", "LAUNCHER_FILE", "localdir", "VENV", "SCRIPT", "DATA_DIR", "DATA_FILE",
    "DATA_PATH", "WORKFLOW", "WORKLOAD_ARGS", "CPUS_PER_TASK", "RUN_ID", "OUTPUT_DIR",
    "WORKERS", "w", "LOG_FILE", "CASE_ARGS",
}
_DARSHAN_VARIABLES = {
    "LD_PRELOAD", "DARSHAN_CONFIG_PATH", "DARSHAN_ENABLE_NONMPI", "DXT_ENABLE_IO_TRACE",
    "DARSHAN_LOGDIR", "DARSHAN_LOGPATH",
}
# Shell and process variables that an experiment argument must not overwrite.
_PROTECTED_VARIABLES = {
    "PATH", "HOME", "IFS", "PWD", "OLDPWD", "CDPATH", "SHELL", "USER", "LOGNAME", "LANG",
    "TERM", "HOSTNAME", "TMPDIR", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "VIRTUAL_ENV", "SECONDS", "RANDOM", "LINENO", "UID", "EUID", "PPID", "OPTIND", "OPTARG", "PS4",
}
_PROTECTED_PREFIXES = ("BASH", "LC_", "SLURM_", "SRUN_", "SBATCH_", "DARSHAN_", "DXT_")

# Schema-v1 builder intentionally does not materialize rank filters.
# config.py validates their syntax, but rank_include/rank_exclude are
# currently out of scope and are silently ignored. The generated config
# retains only the fixed commented Darshan rank example below.
_RANK_EXAMPLE = [
    "# exclude instrumentation for all ranks first",
    "# RANK_EXCLUDE    0:",
    "# then selectively re-include ranks 0-3 and 12:15",
    "# RANK_INCLUDE    0",
    "# RANK_INCLUDE    0:3",
    "# RANK_INCLUDE    12:15",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_campaign(path: str | Path = ".", target: str = "all") -> list[Path]:
    """Validate campaign.yaml and write the requested launchers into its build bundle.

    `target` is "local", "slurm" or "all" (every enabled target). Returns the
    written artifacts. Nothing is written if validation or rendering fails.
    """
    if target != "all" and target not in TARGETS:
        raise DarshanFlowError(f"invalid target: {target}")
    root = require_campaign(path)
    source = (root / CONFIG_FILE).read_bytes()
    config = load_campaign_config(root)
    if (root / CONFIG_FILE).read_bytes() != source:
        raise DarshanFlowError(f"{CONFIG_FILE} changed during the build; run build again")
    targets = _resolve_targets(config, target)
    _check_variable_names(config)
    bundle = bundle_path(root, source)

    # Render every artifact in memory first: any failure here writes nothing.
    artifacts = {CONFIG_FILE: source}
    generated = _generated_config_path(config, root)
    if generated is not None:
        artifacts[generated.as_posix()] = _render_darshan_config(config["darshan"]).encode()
    for name in targets:
        artifacts[f"{name}.sh"] = _render_launcher(name, config, root, bundle).encode()

    print(f"Building campaign: {root}\n")
    print(f"Build bundle:\n  {bundle.relative_to(root)}/\n")
    print("Generated:")
    written = []
    for relative, data in artifacts.items():
        artifact = bundle / relative
        _write_atomic(artifact, data, 0o755 if relative in _BUNDLE_ENTRIES - {CONFIG_FILE} else 0o644)
        written.append(artifact)
        print(f"  {artifact.relative_to(root)}")
    print("\nBuild complete.")
    return written


def bundle_path(root: Path, source: bytes | None = None) -> Path:
    """Return the bundle directory for `source` (default: the current campaign.yaml bytes)."""
    if source is None:
        source = (root / CONFIG_FILE).read_bytes()
    return root / LAUNCHERS_DIR / f"build-{_compute_build_id(source)}"


def launcher_path(root: Path, target: str) -> Path:
    """Return the `target` launcher of the bundle matching the current campaign.yaml."""
    return bundle_path(root) / f"{target}.sh"


# ---------------------------------------------------------------------------
# Build planning and builder-specific validation
# ---------------------------------------------------------------------------

def _compute_build_id(source: bytes) -> str:
    """Return the deterministic build ID of campaign.yaml bytes and this builder."""
    digest = hashlib.sha256(b"\0".join(
        [source, __version__.encode(), BUILD_FORMAT_VERSION.encode()]
    )).hexdigest()
    return digest[:BUILD_ID_LENGTH]


def _resolve_targets(config: dict, target: str) -> list[str]:
    """Return the targets to build, refusing targets disabled in campaign.yaml."""
    enabled = [name for name in TARGETS if config["execution"][name]["enabled"]]
    if target == "all":
        if not enabled:
            raise DarshanFlowError(
                f"no execution target is enabled in {CONFIG_FILE} "
                f"(execution.local.enabled and execution.slurm.enabled are false)"
            )
        return enabled
    if target not in enabled:
        raise DarshanFlowError(f"{target} execution is disabled in {CONFIG_FILE} "
                               f"(execution.{target}.enabled is false)")
    return [target]


def _check_variable_names(config: dict) -> None:
    """Reject configured names that would overwrite variables the launcher relies on."""
    owned = _LAUNCHER_VARIABLES | (_DARSHAN_VARIABLES if config["darshan"]["enabled"] else set())
    environment = config["environment"]
    exported = set(environment["variables"]) if environment else set()
    for name in sorted(exported & owned):
        raise DarshanFlowError(
            f"environment.variables.{name} conflicts with a variable set by the generated launcher"
        )
    for key in config["experiment"] or {}:
        variable = experiment_shell_variable(key)
        if (variable in owned | _PROTECTED_VARIABLES | exported
                or variable.startswith(_PROTECTED_PREFIXES)):
            raise DarshanFlowError(
                f"experiment.{key} would overwrite the shell variable {variable} "
                "in the generated launcher"
            )


def _generated_config_path(config: dict, root: Path) -> Path | None:
    """Return the generated Darshan config path relative to the campaign root.

    Generated files belong to the build bundle, where this relative path is
    mirrored, so it must lie inside the campaign root. None when DarshanFlow
    generates no Darshan config (disabled or external mode).
    """
    darshan = config["darshan"]
    if not darshan["enabled"] or darshan["config"]["mode"] != "generated":
        return None
    path = darshan["config"]["path"]
    if path == root or not path.is_relative_to(root):
        raise DarshanFlowError(
            f"darshan.config.path must be inside the campaign directory in generated mode, "
            f"got {path}\n(generated files are written into the build bundle)"
        )
    relative = path.relative_to(root)
    if relative.parts[0] in _BUNDLE_ENTRIES or relative.name in _RUN_ENTRIES:
        raise DarshanFlowError(
            f"darshan.config.path {relative.as_posix()!r} conflicts with a file "
            "DarshanFlow writes into the build bundle or run directory"
        )
    return relative


def _require_word(value: str, where: str, reason: str, allow_commas: bool = True) -> str:
    """Fail if `value` cannot be written as a single directive field."""
    if any(char.isspace() for char in value) or (not allow_commas and "," in value):
        raise DarshanFlowError(f"{where}: {value!r} cannot be written: {reason}")
    return value


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------

def _render_launcher(target: str, config: dict, root: Path, bundle: Path) -> str:
    """Render the launcher for `target`.

    Local and Slurm share every block; they differ only in the #SBATCH
    directives, the bundle fallback, the setup commands and srun.
    """
    slurm = config["execution"]["slurm"] if target == "slurm" else None
    blocks = [
        _render_header(bundle, slurm),
        _render_locate(bundle, slurm is not None),
        *_render_environment(config["environment"], target, root),
        _render_workload(config, root),
        _render_experiment(config["experiment"]),
        _render_slurm_resources(slurm),
        _render_run_directory(config, root),
        _render_darshan_environment(config, root),
        _render_execution(config, root, slurm is not None),
    ]
    return "\n\n".join("\n".join(block) for block in blocks if block) + "\n"


def _render_header(bundle: Path, slurm: dict | None) -> list[str]:
    lines = ["#!/bin/bash"]
    if slurm:
        for key, option in _SBATCH_OPTIONS:
            value = _require_word(str(slurm[key]), f"execution.slurm.{key}",
                                  "#SBATCH directives cannot contain whitespace")
            lines.append(f"#SBATCH --{option}={value}")
    return lines + [
        "",
        f"# Generated by DarshanFlow {__version__}",
        f"# Build: {bundle.name}",
        f"# Source: {CONFIG_FILE}",
        "# Do not edit this file manually.",
        "",
    ]


def _render_locate(bundle: Path, slurm: bool) -> list[str]:
    """Locate the bundle and campaign root from the launcher's own path.

    Done first, before setup commands that may change the working directory.
    sbatch executes a spooled copy of the script, so slurm.sh falls back to
    the bundle location recorded at build time.
    """
    lines = [
        "# Locate this build bundle and the campaign root (localdir) from the",
        "# launcher itself, never from the working directory or $SLURM_SUBMIT_DIR.",
        'BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'LAUNCHER_FILE="$BUNDLE_DIR/$(basename "${BASH_SOURCE[0]}")"',
    ]
    if slurm:
        lines += [
            f'if [ ! -f "$BUNDLE_DIR/{CONFIG_FILE}" ]; then',
            "    # sbatch runs a spooled copy of this script: use the bundle",
            "    # location recorded at build time.",
            f"    BUNDLE_DIR={shlex.quote(str(bundle))}",
            "fi",
        ]
    return lines + [
        f'if [ ! -f "$BUNDLE_DIR/{CONFIG_FILE}" ]; then',
        '    echo "error: DarshanFlow build bundle not found: $BUNDLE_DIR" >&2',
        "    exit 1",
        "fi",
        'localdir="$(cd "$BUNDLE_DIR/../.." && pwd)"',
    ]


def _render_environment(environment: dict | None, target: str, root: Path) -> list[list[str]]:
    """Render setup commands, venv activation, variable exports and sanity imports."""
    if environment is None:
        return []
    blocks = []
    if environment["setup"][target]:
        blocks.append([
            f"# Site-specific setup (environment.setup.{target}): trusted shell code, verbatim.",
            *environment["setup"][target],
        ])
    if environment["venv"] is not None:
        blocks.append([
            "# Virtual environment",
            f"VENV={_render_path(environment['venv'], root)}",
            'source "$VENV/bin/activate"',
        ])
    if environment["variables"]:
        blocks.append([
            "# Environment variables",
            *(f"export {name}={shlex.quote(value)}"
              for name, value in environment["variables"].items()),
        ])
    if environment["sanity_imports"]:
        blocks.append([
            "# Sanity imports (diagnostic only): interpreter and module locations.",
            'echo "python3 resolved to: $(command -v python3)"',
            "python3 - <<'PY'",
            "import importlib",
            "",
            f"modules = {environment['sanity_imports']!r}",
            "",
            "for name in modules:",
            "    module = importlib.import_module(name)",
            "    print(f\"{name}: {getattr(module, '__file__', '<unknown>')}\")",
            "PY",
        ])
    return blocks


def _render_workload(config: dict, root: Path) -> list[str]:
    dataset = config["dataset"]["path"]
    return [
        "# Workload and dataset",
        f"SCRIPT={_render_path(config['workload']['script'], root)}",
        f"DATA_DIR={_render_path(dataset.parent, root)}",
        f"DATA_FILE={shlex.quote(dataset.name)}",
        'DATA_PATH="$DATA_DIR/$DATA_FILE"',
        f"WORKFLOW={config['dataset']['format']}",
    ]


def _render_experiment(experiment: dict | None) -> list[str]:
    """Render experiment variables and the shared WORKLOAD_ARGS array.

    Each key maps to KEY_NAME and --key-name through config.py. Boolean true
    passes the bare flag; false keeps the variable but omits the flag.
    """
    variables, flags = [], []
    for key, value in (experiment or {}).items():
        variable, flag = experiment_shell_variable(key), experiment_cli_flag(key)
        text = str(value).lower() if isinstance(value, bool) else str(value)
        variables.append(f"{variable}={shlex.quote(text)}")
        if value is True:
            flags.append(flag)
        elif value is not False:
            flags.append(f'{flag} "${variable}"')
    lines = ["# Fixed workload arguments (experiment)", *variables, ""] if variables else []
    return lines + [
        "WORKLOAD_ARGS=(",
        '    --data-path "$DATA_PATH"',
        *(f"    {flag}" for flag in flags),
        ")",
    ]


def _render_slurm_resources(slurm: dict | None) -> list[str]:
    # Same value as the #SBATCH --cpus-per-task directive.
    return [f"CPUS_PER_TASK={slurm['cpus_per_task']}"] if slurm else []


def _render_run_directory(config: dict, root: Path) -> list[str]:
    """Create a fresh run directory and capture its provenance."""
    lines = [
        "# Run directory: always new, never reused or cleaned.",
        'RUN_ID="${WORKFLOW}_$(date +"%Y%m%d_%H%M%S")"',
        'OUTPUT_DIR="$localdir/runs/$RUN_ID"',
        'if [ -e "$OUTPUT_DIR" ]; then',
        '    echo "error: run directory already exists: $OUTPUT_DIR" >&2',
        "    exit 1",
        "fi",
        'mkdir -p "$localdir/runs"',
        'mkdir "$OUTPUT_DIR" "$OUTPUT_DIR/logs"',
    ]
    if config["darshan"]["enabled"]:
        lines.append('mkdir "$OUTPUT_DIR/darshan_logs"')
    lines += [
        "",
        "# Provenance: the bundle snapshot and launcher that produced this run.",
        f'cp "$BUNDLE_DIR/{CONFIG_FILE}" "$OUTPUT_DIR/config.yaml"',
        'cp "$LAUNCHER_FILE" "$OUTPUT_DIR/launcher.sh"',
    ]
    generated = _generated_config_path(config, root)
    if generated is not None:
        lines.append(f"cp {_join_var('BUNDLE_DIR', generated.as_posix())} "
                     f"{_join_var('OUTPUT_DIR', generated.name)}")
    return lines


def _render_darshan_environment(config: dict, root: Path) -> list[str]:
    """Export the Darshan runtime settings; LD_PRELOAD is set per workload case."""
    darshan = config["darshan"]
    if not darshan["enabled"]:
        return []
    generated = _generated_config_path(config, root)
    # A generated config is used from its run-local copy; an external one in place.
    config_path = (_join_var("OUTPUT_DIR", generated.name) if generated is not None
                   else _render_path(darshan["config"]["path"], root))
    return [
        "# Darshan instrumentation",
        f"export DARSHAN_CONFIG_PATH={config_path}",
        "export DARSHAN_ENABLE_NONMPI=1" if darshan["nonmpi"] else "unset DARSHAN_ENABLE_NONMPI",
        "export DXT_ENABLE_IO_TRACE=1" if darshan["dxt"] else "unset DXT_ENABLE_IO_TRACE",
    ]


def _render_execution(config: dict, root: Path, slurm: bool) -> list[str]:
    """Render the workload cases: one per sweep.num_workers value, or exactly one."""
    darshan = config["darshan"]
    sweep = config["sweep"]
    case = "workers_${w}" if sweep else "run"

    body = [f'LOG_FILE="$OUTPUT_DIR/logs/{case}.txt"']
    if darshan["enabled"]:
        body += [
            f'export DARSHAN_LOGDIR="$OUTPUT_DIR/darshan_logs/{case}"',
            'export DARSHAN_LOGPATH="$DARSHAN_LOGDIR"',
            'mkdir "$DARSHAN_LOGDIR"',
        ]
    body += [
        'CASE_ARGS=("${WORKLOAD_ARGS[@]}"' + (' --num-workers "$w"' if sweep else "") + ")",
        'echo "----------------------------------------------------------------"',
        *(['echo "Running with Workers: $w"'] if sweep else []),
        'echo "Log: $LOG_FILE"',
    ]
    if darshan["enabled"]:
        # Preload Darshan only around the workload so that helper commands
        # (mkdir, cp, ...) never produce Darshan logs.
        body.append(f"export LD_PRELOAD={_render_path(darshan['library'], root)}")
    if slurm:
        body += ["srun \\", "    --cpu-bind=cores \\", "    --ntasks=1 \\",
                 '    --cpus-per-task="$CPUS_PER_TASK" \\', '    python3 "$SCRIPT" \\']
    else:
        body.append('python3 "$SCRIPT" \\')
    body += ['    "${CASE_ARGS[@]}" \\', '    2>&1 | tee "$LOG_FILE"']
    if darshan["enabled"]:
        body.append("unset LD_PRELOAD")

    lines = [
        'echo "Starting Test..."',
        'echo "Run ID: $RUN_ID"',
        'echo "Data Path: $DATA_PATH"',
        'echo "Output Directory: $OUTPUT_DIR"',
        'echo "Script: $SCRIPT"',
        "",
    ]
    if sweep:
        lines += [f"WORKERS=({' '.join(str(w) for w in sweep['num_workers'])})",
                  'for w in "${WORKERS[@]}"; do', *(f"    {line}" for line in body), "done"]
    else:
        lines += body
    lines += ["", 'echo "Test Complete."']
    if config["environment"] and config["environment"]["venv"] is not None:
        lines.append("deactivate")
    return lines


# ---------------------------------------------------------------------------
# Generated Darshan configuration
# ---------------------------------------------------------------------------

def _render_darshan_config(darshan: dict) -> str:
    """Render darshan_env.conf, mirroring the project's reference configuration."""
    config = darshan["config"]

    def joined(key: str) -> str:
        where = f"darshan.config.{key}"
        return ",".join(
            _require_word(item, f"{where}[{i}]", "commas and whitespace separate Darshan entries",
                          allow_commas=False)
            for i, item in enumerate(config[key])
        )

    lines = ["# enable DXT modules, which are off by default"]
    if darshan["dxt"]:
        lines.append("MOD_ENABLE      DXT_POSIX,DXT_MPIIO")
    lines += [
        "",
        "# allocate 4096 file records for POSIX and MPI-IO modules",
        "# (darshan only allocates 1024 per-module by default)",
    ]
    for module, count in config["max_records"].items():
        _require_word(module, f"darshan.config.max_records.{module}",
                      "whitespace separates Darshan fields")
        lines.append(f"MAX_RECORDS     {count:<12}{module}")
    lines += [
        "",
        "# the '*' specifier can be used to apply settings for all modules",
        "# in this case, we want all modules to ignore record names",
        '# prefixed with "/home" (i.e., stored in our home directory),',
        '# with a superseding inclusion for files with a ".out" suffix)',
        f"NAME_EXCLUDE    {joined('name_exclude')}       *",
    ]
    if config["name_include"]:
        lines.append(f"NAME_INCLUDE    {joined('name_include')}       *")
    lines += [
        "",
        "# bump up Darshan's default memory usage to 8 MiB",
        f"MODMEM  {config['modmem']}",
        "",
        "# avoid generating logs for git and ls binaries",
        f"APP_EXCLUDE     {joined('app_exclude')}",
    ]
    if config["app_include"]:
        lines.append(f"APP_INCLUDE     {joined('app_include')}")
    lines += [
        "",
        *_RANK_EXAMPLE,
        "",
        # Independent of darshan.dxt and max_records.DXT_POSIX by design.
        "# only retain DXT traces for files that were accessed",
        "# using small I/O ops 20+% of the time",
        f"DXT_SMALL_IO_TRIGGER    {config['dxt_small_io_trigger']}",
    ]
    if config["dump_config"]:
        lines += ["", "DUMP_CONFIG"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shell rendering and file helpers
# ---------------------------------------------------------------------------

def _render_path(path: Path, root: Path) -> str:
    """Render a configured absolute path as a safely quoted shell word.

    Paths lexically inside the campaign root are rendered from $localdir so
    launchers do not freeze the campaign location; others stay absolute.
    """
    if path == root:
        return '"$localdir"'
    if path.is_relative_to(root):
        return _join_var("localdir", path.relative_to(root).as_posix())
    return shlex.quote(str(path))


def _join_var(variable: str, relative: str) -> str:
    """Render "$variable/relative" as one safely quoted shell word."""
    quoted = shlex.quote(relative)
    if quoted == relative:
        return f'"${variable}/{relative}"'
    return f'"${variable}"/{quoted}'


def _write_atomic(path: Path, data: bytes, mode: int) -> None:
    """Write `data` to `path` so that readers never see a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.chmod(mode)
    os.replace(temporary, path)
