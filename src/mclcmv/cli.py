"""Command-line interface (entry point ``mclcmv-cli`` from pyproject.toml).

Each sub-command delegates to the corresponding pipeline script so that the
CLI and the scripts are always in sync.

Usage
-----
mclcmv-cli preprocess --subject sub-06 --session ses-20250828
mclcmv-cli preprocess --all

mclcmv-cli beamform --subject sub-06 --session ses-20250828
mclcmv-cli beamform --all

mclcmv-cli table
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

# Resolve the scripts directory relative to this file:
# src/mclcmv/cli.py → parents[2] = project root
_SCRIPTS: Path = Path(__file__).resolve().parents[2] / "scripts"


def _run(script: str, extra_args: list[str]) -> None:
    """Run a pipeline script as a subprocess and propagate its exit code."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / script)] + extra_args,
        check=False,
    )
    raise SystemExit(result.returncode)


@click.group()
def main() -> None:
    """mcLCMV — anatomy-guided multicluster LCMV beamforming pipeline."""


@main.command("preprocess")
@click.option("--subject", metavar="SUBJECT_ID", default=None,
              help="BIDS subject ID, e.g. sub-06")
@click.option("--session", metavar="SESSION_ID", default=None,
              help="BIDS session ID, e.g. ses-20250828")
@click.option("--all", "run_all", is_flag=True, default=False,
              help="Process all non-skipped sessions in the orientation registry.")
def preprocess(subject: str | None, session: str | None, run_all: bool) -> None:
    """Stage 01 — MRI surface extraction and EHG preprocessing."""
    if run_all:
        _run("01_preprocess.py", ["--all"])
    elif subject and session:
        _run("01_preprocess.py", ["--subject", subject, "--session", session])
    else:
        raise click.UsageError("Provide --subject and --session, or use --all.")


@main.command("beamform")
@click.option("--subject", metavar="SUBJECT_ID", default=None,
              help="BIDS subject ID, e.g. sub-06")
@click.option("--session", metavar="SESSION_ID", default=None,
              help="BIDS session ID, e.g. ses-20250828")
@click.option("--all", "run_all", is_flag=True, default=False,
              help="Process all non-skipped sessions in the orientation registry.")
def beamform(subject: str | None, session: str | None, run_all: bool) -> None:
    """Stage 02 — windowed dual LCMV beamformer (steering + clustering + WOLA)."""
    if run_all:
        _run("02_lcmv.py", ["--all"])
    elif subject and session:
        _run("02_lcmv.py", ["--subject", subject, "--session", session])
    else:
        raise click.UsageError("Provide --subject and --session, or use --all.")


@main.command("table")
def table() -> None:
    """Stage 03 — print the results table for all processed sessions."""
    _run("03_results_table.py", [])


if __name__ == "__main__":
    main()
