"""Command-line interface (entry point ``mclcmv-cli`` from pyproject.toml)."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """mcLCMV — anatomy-guided multicluster LCMV beamforming pipeline."""


@main.command("preprocess")
@click.option("--subject", required=True, help="Subject ID, e.g. sub-02")
@click.option("--session", required=True, help="Session ID, e.g. ses-20250714")
def preprocess(subject: str, session: str) -> None:
    """Stage 01 — EHG + MRI preprocessing (filtering, surface extraction)."""
    raise NotImplementedError


@main.command("forward")
@click.option("--subject", required=True)
@click.option("--session", required=True)
def forward(subject: str, session: str) -> None:
    """Stage 02 — build steering dictionaries from MRI segmentations."""
    raise NotImplementedError


@main.command("cluster")
@click.option("--subject", required=True)
@click.option("--session", required=True)
def cluster(subject: str, session: str) -> None:
    """Stage 03 — DOA clustering and cluster representative steering vectors."""
    raise NotImplementedError


@main.command("beamform")
@click.option("--subject", required=True)
@click.option("--session", required=True)
def beamform(subject: str, session: str) -> None:
    """Stage 04 — windowed LCMV + overlap-add reconstruction."""
    raise NotImplementedError


@main.command("evaluate")
@click.option("--subject", required=True)
@click.option("--session", required=True)
def evaluate(subject: str, session: str) -> None:
    """Stage 05 — separation metrics, leakage diagnostics, figures."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
