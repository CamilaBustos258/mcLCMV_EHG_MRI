"""EHG I/O — BrainVision (.vhdr / .vmrk / .eeg) read/write.

The migrated .vhdr files are missing the `MarkerFile=` field that MNE requires,
so this module reads the binary data and marker files directly from the on-disk
format rather than delegating to mne.io.read_raw_brainvision.
"""

from __future__ import annotations

import configparser
import struct
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_vhdr(vhdr_path: Path) -> dict:
    """Parse a BrainVision .vhdr file and return key recording parameters."""
    # The file starts with a non-INI header line ("BrainVision Data Exchange …")
    # that configparser cannot handle — strip lines before the first section.
    text = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    ini_lines = [ln for ln in text.splitlines() if not ln.startswith("BrainVision")]
    parser = configparser.ConfigParser(strict=False)
    parser.read_string("\n".join(ini_lines))

    ci = parser["Common Infos"]
    n_channels = int(ci["numberofchannels"])
    sampling_interval_us = int(ci["samplinginterval"])
    fs = 1_000_000.0 / sampling_interval_us

    data_file = ci.get("datafile", vhdr_path.stem + ".eeg")
    # The DataFile= value may include a path; keep only the filename
    data_filename = Path(data_file).name

    # Marker file — may or may not be present
    vmrk_filename = ci.get("markerfile", vhdr_path.stem + ".vmrk")
    vmrk_filename = Path(vmrk_filename).name

    binary_format = parser.get("Binary Infos", "binaryformat", fallback="INT_16")

    return {
        "n_channels": n_channels,
        "fs": fs,
        "data_filename": data_filename,
        "vmrk_filename": vmrk_filename,
        "binary_format": binary_format.strip(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_brainvision(vhdr_path: str | Path) -> tuple[np.ndarray, float]:
    """Load a BrainVision EEG recording and return the raw signal matrix.

    Reads the binary .eeg file (INT_16, MULTIPLEXED) directly using the
    parameters encoded in the .vhdr header.  This bypasses the MNE reader,
    which requires a `MarkerFile=` entry that is absent from the migrated files.

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_samples)
    fs   : float — sampling frequency in Hz
    """
    vhdr_path = Path(vhdr_path)
    info = _parse_vhdr(vhdr_path)
    n_channels = info["n_channels"]
    fs = info["fs"]
    binary_format = info["binary_format"]

    eeg_path = vhdr_path.parent / info["data_filename"]
    if not eeg_path.exists():
        # Try same stem with .eeg extension, then .dat (BrainAnalyzer export)
        for ext in (".eeg", ".dat"):
            candidate = vhdr_path.with_suffix(ext)
            if candidate.exists():
                eeg_path = candidate
                break
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG data file not found: {eeg_path}")

    fmt_map = {
        "INT_16": ("h", 2),
        "INT_32": ("i", 4),
        "IEEE_FLOAT_32": ("f", 4),
    }
    dtype_char, bytes_per_sample = fmt_map.get(binary_format, ("h", 2))

    raw_bytes = eeg_path.read_bytes()
    n_samples = len(raw_bytes) // (n_channels * bytes_per_sample)

    # Demultiplex: shape is (n_samples * n_channels,) → (n_channels, n_samples)
    flat = struct.unpack_from(f"<{n_samples * n_channels}{dtype_char}", raw_bytes)
    data = np.array(flat, dtype=np.float64).reshape(n_samples, n_channels).T

    return data, fs


def read_markers(vhdr_path: str | Path) -> pd.DataFrame:
    """Parse the paired .vmrk file for all event markers.

    The .vmrk file is located in the same directory as the .vhdr, with the
    same stem (e.g., sub-06_ses-20251023_acq-EPI_run-01_mrcorrected.vmrk).
    If the .vhdr contains a `MarkerFile=` reference, that is used instead.

    Returns
    -------
    DataFrame with columns:
        sample_index  – integer sample position (1-based in file → 0-based here)
        marker_code   – marker description, e.g. "R255", "R128"
        onset_sec     – onset in seconds from recording start (requires fs)
    """
    vhdr_path = Path(vhdr_path)
    info = _parse_vhdr(vhdr_path)
    fs = info["fs"]

    vmrk_path = vhdr_path.parent / info["vmrk_filename"]
    if not vmrk_path.exists():
        # Try common marker-file extensions (BrainVision .vmrk, BrainAnalyzer .Markers)
        for ext in (".vmrk", ".Markers"):
            candidate = vhdr_path.with_suffix(ext)
            if candidate.exists():
                vmrk_path = candidate
                break
    if not vmrk_path.exists():
        raise FileNotFoundError(f"Marker file not found: {vmrk_path}")

    rows: list[dict] = []
    with open(vmrk_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format A — BrainAnalyzer CSV (migrated .vmrk files):
            #   "Response, R255, 23652, 1, All"
            # Format B — standard BrainVision INI:
            #   "Mk2=Response,R255,23652,1,0"
            try:
                if line.startswith("Mk") and "=" in line:
                    # Format B
                    _, rest = line.split("=", 1)
                    parts = [p.strip() for p in rest.split(",")]
                    if len(parts) < 3:
                        continue
                    description = parts[1]
                    sample_index = int(parts[2]) - 1  # convert to 0-based
                else:
                    # Format A — skip header/metadata lines
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 3:
                        continue
                    try:
                        sample_index = int(parts[2]) - 1  # position field is 3rd column
                    except ValueError:
                        continue  # header row ("Position") or metadata line
                    description = parts[1]  # description is 2nd column, e.g. "R255"

                rows.append({
                    "sample_index": sample_index,
                    "marker_code": description,
                    "onset_sec": sample_index / fs,
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows, columns=["sample_index", "marker_code", "onset_sec"])
