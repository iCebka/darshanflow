"""Run dispatch: execute the build bundle matching the current campaign.

Lifecycle invariant: a run may only dispatch a build bundle matching the
exact current campaign.yaml, DarshanFlow version and builder format version.

The current campaign.yaml bytes are used purely as a build identity, through
builder.bundle_path(). They are never parsed or validated again here, and the
runner never interprets configuration semantics such as target enablement:
it relies on artifacts only. Any change to campaign.yaml, even a comment,
requires a new `build`; restoring the exact previous bytes makes the previous
bundle current again.

The runner only reads the bundle and dispatches its launcher. The launcher
owns the run itself (RUN_ID, runs/<RUN_ID>/, provenance, logs, execution), so
the runner never writes to the bundle or under runs/.

    local  ->  bash <bundle>/local.sh     synchronous, output streams live
    slurm  ->  sbatch <bundle>/slurm.sh   submission only, the job is not tracked
"""

import shlex
import subprocess
from pathlib import Path

from darshanflow import DarshanFlowError
from darshanflow.builder import TARGETS, bundle_path
from darshanflow.campaign import CONFIG_FILE, require_campaign

LAUNCH_PROGRAM = {"local": "bash", "slurm": "sbatch"}


def run_campaign(path: str | Path, target: str, dry_run: bool = False) -> None:
    """Dispatch the `target` launcher of the bundle matching the current campaign.yaml.

    With `dry_run`, perform every identity and artifact check, but write and
    execute nothing.
    """
    if target not in TARGETS:
        raise DarshanFlowError(f"invalid target: {target}")
    root = require_campaign(path)
    # Read once: these bytes are the single configuration identity of this dispatch.
    source = (root / CONFIG_FILE).read_bytes()
    bundle = bundle_path(root, source)
    launcher = _resolve_launcher(root, source, bundle, target)
    command = [LAUNCH_PROGRAM[target], str(launcher)]

    # The bundle matched `source`; refuse to dispatch it if the campaign changed since.
    if (root / CONFIG_FILE).read_bytes() != source:
        raise DarshanFlowError(f"{CONFIG_FILE} changed while preparing the run; run the command again")

    if dry_run:
        print("Dry run: no command was executed.")
        print(f"Target: {target}")
        print(f"Build bundle:\n  {bundle.relative_to(root)}")
        print(f"Launcher:\n  {launcher.relative_to(root)}\n")
        print(f"Would execute:\n  {shlex.join(command)}")
        return

    print(f"Running campaign: {root}")
    print(f"Target: {target}")
    print(f"Build bundle: {bundle.relative_to(root)}")
    print(f"Launcher: {launcher.relative_to(root)}\n")
    # Flush so that this header precedes the launcher's own output.
    print(f"Executing:\n  {shlex.join(command)}", flush=True)
    _execute(command, root, target)
    print("\nRun complete." if target == "local" else "\nSubmission complete.")


def _resolve_launcher(root: Path, source: bytes, bundle: Path, target: str) -> Path:
    """Return the `target` launcher of `bundle` after checking the bundle's identity.

    Distinguishes a missing bundle (the current configuration was never
    built) from a missing launcher (this target was not built into it).
    """
    name = bundle.relative_to(root)
    if not bundle.is_dir():
        raise DarshanFlowError(
            f"no build bundle matches the current {CONFIG_FILE} (expected {name})\n"
            f"{CONFIG_FILE} or the DarshanFlow/builder version may have changed since the last build.\n"
            "Run `build` again before running the campaign."
        )
    snapshot = bundle / CONFIG_FILE
    if not snapshot.is_file():
        raise DarshanFlowError(
            f"build bundle {name} is incomplete: its {CONFIG_FILE} snapshot is missing\n"
            "Run `build` again to regenerate it."
        )
    if snapshot.read_bytes() != source:
        raise DarshanFlowError(
            f"build bundle {name} does not correspond to the current {CONFIG_FILE} "
            "(its snapshot differs)\nRun `build` again to regenerate it."
        )
    launcher = bundle / f"{target}.sh"
    if not launcher.is_file():
        raise DarshanFlowError(
            f"{target} launcher not found in build bundle {bundle.name}\n"
            f"run `build --target {target}` first"
        )
    return launcher


def _execute(command: list[str], root: Path, target: str) -> None:
    """Run `command` from the campaign root with inherited stdout/stderr."""
    program = command[0]
    action = "local launcher" if target == "local" else "slurm submission"
    try:
        result = subprocess.run(command, cwd=root)
    except FileNotFoundError:
        raise DarshanFlowError(f"cannot execute {target} target: {program} was not found") from None
    except OSError as exc:
        raise DarshanFlowError(f"cannot execute {target} target: {program}: {exc.strerror}") from None
    except KeyboardInterrupt:
        raise DarshanFlowError(f"{action} interrupted") from None
    if result.returncode < 0:
        raise DarshanFlowError(f"{action} terminated by signal {-result.returncode}")
    if result.returncode != 0:
        raise DarshanFlowError(f"{action} failed with exit code {result.returncode}")
