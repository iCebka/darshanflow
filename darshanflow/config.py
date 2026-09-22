"""Campaign configuration: loading, validation and normalization (schema v1).

This module is the single authority on what campaign.yaml means. Other
modules must not open or interpret campaign.yaml themselves; they consume the
normalized configuration returned by load_campaign_config().

Validation is strict: unknown keys, unsupported values and malformed types
raise DarshanFlowError naming the offending location as a dotted path (e.g.
execution.slurm.nodes). Every section is closed-schema except `experiment`,
which is deliberately open (see _experiment()).

Schema v1 defaults structure, never behavior: every behavioral decision
(darshan.enabled, execution.*.enabled, analysis toggles, ...) must be written
explicitly, and an incomplete configuration is an error. Three kinds of
fields exist:

  - always required: must be present and non-null;
  - optional (experiment, sweep, environment and the children of
    environment): absent, null or empty means "not used";
  - conditionally required: the settings of a feature become required once
    it is enabled (darshan.enabled, execution.slurm.enabled). While the
    feature is disabled, supplied values are type-checked, then dropped.

The normalized configuration is a fresh plain nested dict mirroring the YAML
layout. Unused optional sections and the settings of disabled features are
None. Path fields become absolute, lexically normalized pathlib.Path objects
(symlinks not resolved); relative paths are relative to the campaign root,
never to the current working directory. Paths are not required to exist:
existence is checked by the stages that use them.
"""

import difflib
import os
import re
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml

from darshanflow import DarshanFlowError
from darshanflow.campaign import CONFIG_FILE

# Version of the campaign.yaml schema; independent of the manifest schema.
SUPPORTED_CONFIG_SCHEMA_VERSION = 1
WORKLOAD_TYPES = ("python",)
DATASET_FORMATS = ("hdf5", "csv", "npz", "bin", "root")
DARSHAN_CONFIG_MODES = ("generated", "external")
EXECUTION_TARGETS = ("local", "slurm")
ANALYSIS_METRICS = ("io", "access", "metadata", "bandwidth", "balance")
ANALYSIS_GRAPHS = ("compact", "individual")

# Experiment keys managed structurally elsewhere, mapped to their replacement.
RESERVED_EXPERIMENT_KEYS = {"data_path": "dataset.path", "num_workers": "sweep.num_workers"}

_TOP_LEVEL_KEYS = (
    "schema_version", "campaign", "workload", "dataset", "experiment",
    "sweep", "environment", "darshan", "execution", "analysis",
)
_SLURM_KEYS = ("enabled", "nodes", "cpus_per_task", "time", "constraint", "qos", "account")
# darshan.config keys that only apply when DarshanFlow generates the file.
_GENERATED_ONLY_KEYS = (
    "max_records", "name_exclude", "name_include", "modmem", "app_exclude",
    "app_include", "rank_exclude", "rank_include", "dxt_small_io_trigger", "dump_config",
)

# Single underscores only, so key -> KEY -> --key conversion is unambiguous.
_EXPERIMENT_KEY = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_ENV_VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PYTHON_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_RANK = re.compile(r"^([0-9]+)(?::([0-9]+))?$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_campaign_config(campaign: str | Path) -> dict[str, Any]:
    """Load, validate and normalize campaign.yaml from the campaign root `campaign`."""
    root = Path(campaign).resolve()
    return load_config_file(root / CONFIG_FILE, root)


def load_config_file(path: str | Path, campaign_root: str | Path) -> dict[str, Any]:
    """Load, validate and normalize a schema-v1 YAML file, e.g. a run's config.yaml snapshot.

    Same loader and validation as campaign.yaml; relative paths resolve
    against `campaign_root`.
    """
    root = Path(campaign_root).resolve()
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = yaml.load(stream, Loader=_StrictLoader)
    except OSError as exc:
        raise DarshanFlowError(f"cannot read {path}: {exc.strerror}") from None
    except yaml.YAMLError as exc:
        raise DarshanFlowError(f"invalid YAML in {path}:\n{exc}") from None
    try:
        return validate_campaign_config(raw, root)
    except DarshanFlowError as exc:
        raise DarshanFlowError(f"invalid campaign configuration ({path}):\n{exc}") from None


def validate_campaign_config(raw: Any, root: str | Path) -> dict[str, Any]:
    """Validate parsed campaign.yaml content and return its normalized form.

    `raw` is the object produced by the YAML parser and is never modified.
    `root` is the campaign directory that relative paths are resolved against.
    """
    root = Path(root).resolve()
    if raw is None:
        raise DarshanFlowError("configuration is empty")
    if not isinstance(raw, dict):
        raise DarshanFlowError(f"configuration must be a mapping, got {_got(raw)}")

    # Check the version first: other versions may legitimately use other keys.
    if "schema_version" not in raw:
        raise DarshanFlowError("missing required configuration key: schema_version")
    version = raw["schema_version"]
    if type(version) is not int or version != SUPPORTED_CONFIG_SCHEMA_VERSION:
        raise DarshanFlowError(
            f"unsupported campaign schema version: {version!r}\n"
            f"supported schema version: {SUPPORTED_CONFIG_SCHEMA_VERSION}"
        )

    top = _section(raw, "", _TOP_LEVEL_KEYS)
    return {
        "schema_version": version,
        "campaign": _get(top, "", "campaign", _campaign),
        "workload": _get(top, "", "workload", _workload, root),
        "dataset": _get(top, "", "dataset", _dataset, root),
        # Optional features: an omitted, null or empty section becomes None.
        "experiment": _get(top, "", "experiment", _experiment, required=False),
        "sweep": _get(top, "", "sweep", _sweep, required=False),
        "environment": _get(top, "", "environment", _environment, root, required=False),
        "darshan": _get(top, "", "darshan", _darshan, root),
        "execution": _get(top, "", "execution", _execution),
        "analysis": _get(top, "", "analysis", _analysis),
    }


def experiment_shell_variable(key: str) -> str:
    """Return the launcher shell variable of an experiment key (batch_size -> BATCH_SIZE)."""
    return key.upper()


def experiment_cli_flag(key: str) -> str:
    """Return the workload CLI flag of an experiment key (batch_size -> --batch-size)."""
    return "--" + key.replace("_", "-")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _campaign(value: Any, where: str) -> dict:
    sec = _section(value, where, ("name",))
    return {"name": _get(sec, where, "name", _string)}


def _workload(value: Any, where: str, root: Path) -> dict:
    sec = _section(value, where, ("type", "script"))
    return {
        "type": _get(sec, where, "type", _choice, WORKLOAD_TYPES, "workload type"),
        "script": _get(sec, where, "script", _path, root),
    }


def _dataset(value: Any, where: str, root: Path) -> dict:
    """Validate `dataset`. The format is never inferred from the path."""
    sec = _section(value, where, ("format", "path"))
    return {
        "format": _get(sec, where, "format", _choice, DATASET_FORMATS, "dataset format"),
        "path": _get(sec, where, "path", _path, root),
    }


def _experiment(value: Any, where: str) -> dict | None:
    """Validate the open-schema `experiment` section; empty means None.

    Unlike every other section, `experiment` accepts arbitrary keys: they are
    workload CLI arguments whose meaning DarshanFlow deliberately does not
    know or validate. Each key maps mechanically to a shell variable and a
    CLI flag (batch_size -> BATCH_SIZE -> --batch-size). Values must be
    scalars; a boolean true passes the bare flag and false omits it. Null
    values are errors: an unwanted argument must be omitted instead.
    """
    result = {}
    for key, item in _section(value, where, None).items():
        key_where = _join(where, key)
        if key in RESERVED_EXPERIMENT_KEYS:
            raise DarshanFlowError(
                f"{key_where} is managed by DarshanFlow: use {RESERVED_EXPERIMENT_KEYS[key]} instead"
            )
        if not isinstance(key, str) or not _EXPERIMENT_KEY.match(key):
            raise DarshanFlowError(
                f"invalid experiment key: {key!r}\n"
                "experiment keys must be lowercase snake_case, e.g. batch_size"
            )
        if not isinstance(item, (str, int, float)):  # bool is an int subclass
            raise DarshanFlowError(
                f"{key_where} must be a string, integer, float or boolean, got {_got(item)}"
            )
        result[key] = item
    return result or None


def _sweep(value: Any, where: str) -> dict | None:
    """Validate `sweep`; empty means None: run once without --num-workers."""
    sec = _section(value, where, ("num_workers",))
    if not sec:
        return None
    workers_where = f"{where}.num_workers"
    items = _list(sec["num_workers"], workers_where)
    if not items:
        raise DarshanFlowError(f"{workers_where} must not be empty")
    workers = [_int(v, f"{workers_where}[{i}]", positive=False) for i, v in enumerate(items)]
    duplicates = sorted({v for v in workers if workers.count(v) > 1})
    if duplicates:
        raise DarshanFlowError(f"{workers_where} contains duplicate values: {duplicates}")
    return {"num_workers": workers}


def _environment(value: Any, where: str, root: Path) -> dict | None:
    """Validate `environment`; every child is optional.

    A section that configures nothing is None. Otherwise omitted or null
    children normalize to "nothing": venv None and empty containers.

    environment.setup entries are trusted shell code: an escape hatch for
    site-specific setup, inserted verbatim into generated launchers in this
    order: setup commands, venv activation, variable exports, sanity imports,
    Darshan environment, workload. sanity_imports are diagnostic only; they
    are neither installed nor imported by DarshanFlow itself.
    """
    sec = _section(value, where, ("venv", "variables", "setup", "sanity_imports"))
    setup = _get(sec, where, "setup", _section, EXECUTION_TARGETS, required=False) or {}
    result = {
        "venv": _get(sec, where, "venv", _path, root, required=False),
        "variables": _get(sec, where, "variables", _variables, required=False) or {},
        "setup": {
            target: _get(setup, f"{where}.setup", target, _string_list, required=False) or []
            for target in EXECUTION_TARGETS
        },
        "sanity_imports": _get(
            sec, where, "sanity_imports", _string_list, required=False,
            pattern=_PYTHON_MODULE, label="Python module name", unique=True,
        ) or [],
    }
    unused = (result["venv"] is None and not result["variables"]
              and not any(result["setup"].values()) and not result["sanity_imports"])
    return None if unused else result


def _variables(value: Any, where: str) -> dict[str, str]:
    """Validate environment variables; values are normalized to strings."""
    result = {}
    for name, item in _section(value, where, None).items():
        if not isinstance(name, str) or not _ENV_VAR.match(name):
            raise DarshanFlowError(f"invalid environment variable name: {name!r}")
        if isinstance(item, bool):
            result[name] = str(item).lower()
        elif isinstance(item, (str, int, float)):
            result[name] = str(item)
        else:
            raise DarshanFlowError(
                f"{_join(where, name)} must be a string, number or boolean, got {_got(item)}"
            )
    return result


def _darshan(value: Any, where: str, root: Path) -> dict:
    """Validate `darshan`; `enabled` is always required.

    When enabled, library, nonmpi, dxt and config are required. When
    disabled, supplied values are only type-checked and every setting
    normalizes to None: no Darshan configuration is inferred.
    """
    sec = _section(value, where, ("enabled", "library", "nonmpi", "dxt", "config"))
    enabled = _get(sec, where, "enabled", _bool)
    opts = {"required": enabled, "condition": f"{where}.enabled is true"}
    settings = {
        "library": _get(sec, where, "library", _path, root, **opts),
        "nonmpi": _get(sec, where, "nonmpi", _bool, **opts),
        "dxt": _get(sec, where, "dxt", _bool, **opts),
        "config": _get(sec, where, "config", _darshan_config, root, enabled, **opts),
    }
    return {"enabled": enabled, **(settings if enabled else dict.fromkeys(settings))}


def _darshan_config(value: Any, where: str, root: Path, strict: bool) -> dict:
    """Validate `darshan.config` in generated or external mode.

    With `strict` (Darshan enabled) every field of the selected mode is
    required; otherwise supplied values are only checked. In generated mode
    DarshanFlow promises a complete darshan_env.conf, so every decision it
    can generate must be explicit. darshan.dxt, dxt_small_io_trigger and a
    DXT_POSIX entry in max_records are independent: no cross-field rules.
    """
    sec = _section(value, where, ("mode", "path", *_GENERATED_ONLY_KEYS))
    opts = {"required": strict, "condition": "darshan.enabled is true"}
    mode = _get(sec, where, "mode", _choice, DARSHAN_CONFIG_MODES, "Darshan config mode", **opts)
    path = _get(sec, where, "path", _path, root, **opts)

    if mode == "external":
        for key in _GENERATED_ONLY_KEYS:
            if key in sec:
                raise DarshanFlowError(
                    f"{where}.{key} is only supported when {where}.mode is 'generated'"
                )
        return {"mode": mode, "path": path}

    def get(key: str, check, **kwargs) -> Any:
        return _get(sec, where, key, check, required=strict,
                    condition=f"{where}.mode is 'generated'", **kwargs)

    return {
        "mode": mode,
        "path": path,
        "max_records": get("max_records", _max_records),
        # Name filters are Darshan regex patterns, not filesystem paths.
        "name_exclude": get("name_exclude", _string_list, non_empty=True),
        "name_include": get("name_include", _string_list),
        "modmem": get("modmem", _int),
        "app_exclude": get("app_exclude", _string_list, non_empty=True),
        "app_include": get("app_include", _string_list),
        "rank_exclude": get("rank_exclude", _rank_list),
        "rank_include": get("rank_include", _rank_list),
        "dxt_small_io_trigger": get("dxt_small_io_trigger", _fraction),
        "dump_config": get("dump_config", _bool),
    }


def _max_records(value: Any, where: str) -> dict[str, int]:
    """Map Darshan module names to positive record limits; at least one entry."""
    sec = _section(value, where, None)
    if not sec:
        raise DarshanFlowError(f"{where} must contain at least one module")
    result = {}
    for module, count in sec.items():
        if not isinstance(module, str) or not module.strip():
            raise DarshanFlowError(f"invalid Darshan module name in {where}: {module!r}")
        result[module] = _int(count, _join(where, module))
    return result


def _execution(value: Any, where: str) -> dict:
    """Validate `execution`; both targets and their `enabled` flags are required.

    The Slurm settings are required when Slurm is enabled. When disabled,
    supplied values are only type-checked and normalize to None.
    """
    sec = _section(value, where, EXECUTION_TARGETS)
    local_where, slurm_where = f"{where}.local", f"{where}.slurm"
    local = _get(sec, where, "local", _section, ("enabled",))
    slurm = _get(sec, where, "slurm", _section, _SLURM_KEYS)

    enabled = _get(slurm, slurm_where, "enabled", _bool)
    opts = {"required": enabled, "condition": f"{slurm_where}.enabled is true"}
    settings = {
        key: _get(slurm, slurm_where, key, check, **opts)
        for key, check in (("nodes", _int), ("cpus_per_task", _int), ("time", _string),
                           ("constraint", _string), ("qos", _string), ("account", _string))
    }
    return {
        "local": {"enabled": _get(local, local_where, "enabled", _bool)},
        "slurm": {"enabled": enabled, **(settings if enabled else dict.fromkeys(settings))},
    }


def _analysis(value: Any, where: str) -> dict:
    """Validate `analysis`; every metric and graph toggle is required.

    Both graph families disabled means no graphs; both enabled means both.
    Explicit CLI options will later override these values.
    """
    sec = _section(value, where, ("metrics", "graphs"))
    return {
        "metrics": _get(sec, where, "metrics", _toggles, ANALYSIS_METRICS),
        "graphs": _get(sec, where, "graphs", _toggles, ANALYSIS_GRAPHS),
    }


def _toggles(value: Any, where: str, names: tuple[str, ...]) -> dict:
    """Validate a mapping of every one of `names` to an {enabled: bool} entry."""
    sec = _section(value, where, names)
    result = {}
    for name in names:
        entry = _get(sec, where, name, _section, ("enabled",))
        result[name] = {"enabled": _get(entry, _join(where, name), "enabled", _bool)}
    return result


# ---------------------------------------------------------------------------
# Generic validators: each takes (value, where, ...) and returns the value
# ---------------------------------------------------------------------------

def _section(value: Any, where: str, allowed: tuple[str, ...] | None) -> dict:
    """Return `value` as a mapping; unknown keys are errors unless `allowed` is None."""
    if not isinstance(value, dict):
        raise DarshanFlowError(f"{where} must be a mapping, got {_got(value)}")
    if allowed is not None:
        for key in value:
            if key not in allowed:
                message = f"unknown configuration key: {_join(where, key)}"
                close = difflib.get_close_matches(str(key), allowed, n=1)
                if close:
                    message += f"\ndid you mean: {_join(where, close[0])}"
                raise DarshanFlowError(message)
    return value


def _get(sec: dict, where: str, key: str, check, *args, required: bool = True,
         condition: str | None = None, **kwargs) -> Any:
    """Return check(sec[key], "where.key", *args, **kwargs).

    Distinguishes a missing key, an explicit null and a concrete value. A
    required key must be present, and an explicit null reaches `check`,
    which rejects it. An optional key that is missing or null yields None.
    `condition` explains why a conditionally required key is required.
    """
    if key in sec and (sec[key] is not None or required):
        return check(sec[key], _join(where, key), *args, **kwargs)
    if not required:
        return None
    reason = f" (required when {condition})" if condition else ""
    raise DarshanFlowError(f"missing required configuration key: {_join(where, key)}{reason}")


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DarshanFlowError(f"{where} must be a non-empty string, got {_got(value)}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise DarshanFlowError(f"{where} must be a boolean (true or false), got {_got(value)}")
    return value


def _int(value: Any, where: str, positive: bool = True) -> int:
    """Return a positive (or non-negative) integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        kind = "positive" if positive else "non-negative"
        raise DarshanFlowError(f"{where} must be a {kind} integer, got {_got(value)}")
    return value


def _fraction(value: Any, where: str) -> float:
    """Return a number between 0 and 1 inclusive, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise DarshanFlowError(f"{where} must be a number between 0 and 1, got {_got(value)}")
    return float(value)


def _choice(value: Any, where: str, choices: tuple[str, ...], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise DarshanFlowError(
            f"{where}: unsupported {label} {value!r}\n"
            f"supported {label}s: {', '.join(choices)}"
        )
    return value


def _path(value: Any, where: str, root: Path) -> Path:
    """Return an absolute path; relative paths are relative to the campaign `root`."""
    text = _string(value, where)
    if "$" in text:
        raise DarshanFlowError(
            f"{where}: environment variables are not expanded in paths, got {text!r}"
        )
    return Path(os.path.normpath(root / Path(text).expanduser()))


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise DarshanFlowError(f"{where} must be a list, got {_got(value)}")
    return value


def _string_list(value: Any, where: str, pattern: re.Pattern | None = None,
                 label: str = "value", unique: bool = False, non_empty: bool = False) -> list[str]:
    """Return a list of non-empty strings, optionally matching `pattern` and unique.

    With `non_empty`, the list itself must contain at least one entry.
    """
    items = _list(value, where)
    if non_empty and not items:
        raise DarshanFlowError(f"{where} must contain at least one entry")
    for i, item in enumerate(items):
        _string(item, f"{where}[{i}]")
        if pattern and not pattern.match(item):
            raise DarshanFlowError(f"{where}[{i}]: invalid {label} {item!r}")
        if unique and item in items[:i]:
            raise DarshanFlowError(f"{where}: duplicate {label} {item!r}")
    return list(items)


def _rank_list(value: Any, where: str) -> list[str]:
    """Return Darshan rank filters (N or N:M, N <= M) normalized to strings."""
    result = []
    for i, item in enumerate(_list(value, where)):
        if not isinstance(item, bool) and isinstance(item, int) and item >= 0:
            result.append(str(item))
            continue
        match = _RANK.match(item) if isinstance(item, str) else None
        if not match or (match[2] is not None and int(match[1]) > int(match[2])):
            raise DarshanFlowError(
                f"{where}[{i}]: invalid rank expression {item!r} (expected N or N:M with N <= M)"
            )
        result.append(":".join(str(int(part)) for part in match.groups() if part is not None))
    return result


def _join(where: str, key: Any) -> str:
    return f"{where}.{key}" if where else str(key)


def _got(value: Any) -> str:
    """Describe `value` for error messages using YAML vocabulary."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"boolean {str(value).lower()}"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, list):
        return "a list"
    kind = {int: "integer", float: "float", str: "string"}.get(type(value), type(value).__name__)
    return f"{kind} {value!r}"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys and does not read N:M as base 60."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, Hashable):
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"duplicate key {key!r}", key_node.start_mark
                    )
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


# YAML 1.1 reads unquoted values such as 12:15 as base-60 numbers (735),
# silently corrupting rank ranges and Slurm times. As in YAML 1.2, resolve
# them as plain strings instead: this resolver is checked before int/float.
_BASE60 = re.compile(r"^[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+(?:\.[0-9_]*)?$")
_StrictLoader.yaml_implicit_resolvers = {
    first: list(resolvers) for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _first in "-+0123456789":
    _StrictLoader.yaml_implicit_resolvers.setdefault(_first, []).insert(
        0, ("tag:yaml.org,2002:str", _BASE60)
    )
