#!/usr/bin/env python3
"""DarshanFlow command-line interface.

This module only parses arguments and dispatches to the lifecycle modules:
init -> build -> run -> analyze. Stage logic lives in the darshanflow package.
"""

import argparse
import sys

from darshanflow import DarshanFlowError, __version__
from darshanflow.analysis import GRAPH_MODES, METRICS, analyze_campaign
from darshanflow.builder import TARGETS, build_campaign
from darshanflow.campaign import init_campaign
from darshanflow.runner import run_campaign

CAMPAIGN_HELP = "Campaign directory (default: current directory)."


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with one subcommand per lifecycle stage."""
    parser = argparse.ArgumentParser(
        description="DarshanFlow: configure, build, run and analyze "
        "Darshan-instrumented ML I/O experiment campaigns."
    )
    parser.add_argument("--version", action="version", version=f"DarshanFlow {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, summary: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=summary, description=summary)

    p = add("init", "Create a new DarshanFlow campaign.")
    p.add_argument("campaign", metavar="CAMPAIGN", help="Path of the campaign to create.")
    p.set_defaults(func=lambda a: init_campaign(a.campaign))

    p = add("build", "Generate execution artifacts from campaign configuration.")
    p.add_argument("campaign", nargs="?", default=".", metavar="CAMPAIGN", help=CAMPAIGN_HELP)
    p.add_argument("--target", choices=[*TARGETS, "all"], default="all",
                   help="Launcher(s) to generate (default: all).")
    p.set_defaults(func=lambda a: build_campaign(a.campaign, a.target))

    p = add("run", "Prepare and execute a campaign run.")
    p.add_argument("campaign", nargs="?", default=".", metavar="CAMPAIGN", help=CAMPAIGN_HELP)
    p.add_argument("--target", choices=TARGETS, required=True, help="Execution target.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be executed without writing or executing anything.")
    p.set_defaults(func=lambda a: run_campaign(a.campaign, a.target, a.dry_run))

    p = add("analyze", "Prepare and run post-experiment analysis.")
    p.add_argument("campaign", nargs="?", default=".", metavar="CAMPAIGN", help=CAMPAIGN_HELP)
    p.add_argument("--run", default="latest", metavar="RUN_ID",
                   help="Run to analyze, or 'latest' (default: latest).")
    # None means "not given": the run's config.yaml snapshot then decides.
    p.add_argument("--metrics", nargs="+", choices=[*METRICS, "all"], default=None,
                   metavar="METRIC",
                   help=f"Metrics to compute: {', '.join(METRICS)}, or all (every metric "
                        "enabled in the run's config.yaml). Default: the run's enabled metrics.")
    p.add_argument("--graphs", choices=GRAPH_MODES, default=None,
                   help="Graphs to generate: none, a mode, or all (every mode enabled in the "
                        "run's config.yaml). Default: the run's enabled modes.")
    p.set_defaults(func=lambda a: analyze_campaign(a.campaign, a.run, a.metrics, a.graphs))

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch the command and report user-facing errors."""
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (DarshanFlowError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
