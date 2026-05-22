"""Stage 01 — session preprocessing: MRI surface extraction + EHG trimming.

Reads BIDS-organised source data (set MCLCMV_SOURCEDATA env var or place data
under data/sourcedata/), applies per-session MRI orientation correction, extracts organ surfaces
and electrode centroids, trims the EHG to the MRI acquisition window, and saves
the result to data/derivatives/synced/.

The output session_data.npz is the only input required by 02_lcmv.py — the two
stages can be run independently after the first stage completes.

Usage
-----
# Single session
python scripts/01_preprocess.py --subject sub-06 --session ses-20250828

# All registered sessions
python scripts/01_preprocess.py --all
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

# ── Project root on sys.path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mclcmv.config import (
    DBSCAN_EPS_CM,
    DBSCAN_MIN_SAMPLES,
    DERIV_SYNCED,
    EXPECTED_N_ELECTRODES,
    MAX_SURFACE_VERTS,
    SOURCEDATA,
    load_orientation_registry,
)
from mclcmv.io.ehg import load_brainvision, read_markers
from mclcmv.io.mri import load_session_volumes
from mclcmv.preprocessing.ehg import clip_amplitudes, find_mri_trim_indices
from mclcmv.preprocessing.mri import (
    extract_electrode_coords,
    extract_surface,
    subsample_surface,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


def _find_ehg_vhdr(ehg_dir: Path) -> Path:
    """Return the EPI acquisition .vhdr file from a BIDS EHG folder.

    Preference: acq-EPI run-01 > acq-EPI any run > any .vhdr.
    Skips macOS resource-fork shadow files (._…).
    """
    for pattern in (
        "*_acq-EPI_run-01_mrcorrected.vhdr",
        "*_acq-EPI_*_mrcorrected.vhdr",
        "*_acq-EPI_*.vhdr",
        "*.vhdr",
    ):
        candidates = sorted(
            p for p in ehg_dir.glob(pattern) if not p.name.startswith("._")
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No .vhdr file found in {ehg_dir}")


def preprocess_session(subject_id: str, session_id: str) -> Path:
    """Preprocess one session and save output to derivatives/synced/.

    Returns
    -------
    Path to the output directory.
    """
    registry = load_orientation_registry()

    if subject_id not in registry:
        raise KeyError(f"{subject_id} not in orientation registry.")
    if session_id not in registry[subject_id]:
        raise KeyError(f"{session_id} not registered for {subject_id}.")

    meta = registry[subject_id][session_id]
    orientation_case = meta["orientation_case"]
    sex = meta["sex"]
    cycle_phase = meta["cycle_phase"]

    session_sourcedata = SOURCEDATA / subject_id / session_id
    ehg_dir = session_sourcedata / "ehg"
    mri_dir = session_sourcedata / "mri"

    # ── EHG ───────────────────────────────────────────────────────────────
    vhdr_path = _find_ehg_vhdr(ehg_dir)
    log.info("EHG: %s", vhdr_path.name)
    ehg_data, fs = load_brainvision(vhdr_path)
    markers_df = read_markers(vhdr_path)
    trim_start, trim_end = find_mri_trim_indices(markers_df)
    n_trimmed = trim_end - trim_start
    log.info(
        "  MRI window: samples %d–%d  (%d samples = %.1f min)",
        trim_start, trim_end, n_trimmed, n_trimmed / fs / 60,
    )

    ehg_trimmed = ehg_data[:, trim_start:trim_end]
    ehg_clipped = clip_amplitudes(ehg_trimmed, clip_factor=10.0)

    # ── MRI ───────────────────────────────────────────────────────────────
    log.info("MRI orientation: %s", orientation_case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volumes = load_session_volumes(mri_dir, sex, orientation_case)

    voxel_sizes = volumes["voxel_sizes"]
    log.info("  Voxel sizes: %.2f × %.2f × %.2f mm", *voxel_sizes)

    # Organ surfaces
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bladder_verts, bladder_faces = extract_surface(
            volumes["bladder_mask"], voxel_sizes)

    if volumes["uterus_mask"] is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uterus_verts, uterus_faces = extract_surface(
                volumes["uterus_mask"], voxel_sizes)
    else:
        uterus_verts = np.zeros((0, 3), dtype=float)
        uterus_faces = np.zeros((0, 3), dtype=np.intp)

    uterus_verts, uterus_faces   = subsample_surface(
        uterus_verts,  uterus_faces,  max_verts=MAX_SURFACE_VERTS)
    bladder_verts, bladder_faces = subsample_surface(
        bladder_verts, bladder_faces, max_verts=MAX_SURFACE_VERTS)

    log.info(
        "  Surfaces — uterus: %d verts, bladder: %d verts",
        len(uterus_verts), len(bladder_verts),
    )

    # Electrode centroids
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        electrode_coords = extract_electrode_coords(
            volumes["electrodes_mask"],
            voxel_sizes,
            expected_n=EXPECTED_N_ELECTRODES,
            eps_cm=DBSCAN_EPS_CM,
            min_samples=DBSCAN_MIN_SAMPLES,
        )

    log.info("  Electrodes: %d centroids found.", len(electrode_coords))

    # Sort electrodes by Z then X for reproducible channel ordering
    sort_idx = np.lexsort((electrode_coords[:, 0], electrode_coords[:, 2]))
    electrode_coords = electrode_coords[sort_idx]

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = DERIV_SYNCED / subject_id / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "session_data.npz",
        ehg_data        = ehg_clipped,      # (n_channels, n_mri_samples) trimmed+clipped
        sampling_rate   = np.array(fs),
        electrode_coords= electrode_coords, # (n_electrodes, 3) cm
        uterus_verts    = uterus_verts,     # (G_U, 3) cm
        uterus_faces    = uterus_faces,
        bladder_verts   = bladder_verts,    # (G_B, 3) cm
        bladder_faces   = bladder_faces,
        voxel_sizes     = voxel_sizes,      # (3,) mm — kept for reference
    )

    meta_out = {
        "subject_id":      subject_id,
        "session_id":      session_id,
        "sex":             sex,
        "cycle_phase":     cycle_phase,
        "orientation_case":orientation_case,
        "vhdr_source":     vhdr_path.name,
        "n_channels":      int(ehg_data.shape[0]),
        "n_mri_samples":   int(n_trimmed),
        "sampling_rate_hz":float(fs),
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta_out, fh, indent=2)

    log.info("Saved → %s", out_dir)
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 01: preprocess MRI + EHG for one or all sessions."
    )
    parser.add_argument("--subject", metavar="SUBJECT_ID",
                        help="BIDS subject, e.g. sub-06")
    parser.add_argument("--session", metavar="SESSION_ID",
                        help="BIDS session, e.g. ses-20250828")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Process all sessions in the orientation registry.")
    args = parser.parse_args()

    if not args.run_all and (args.subject is None or args.session is None):
        parser.error("Provide --subject and --session, or use --all.")

    registry = load_orientation_registry()

    if args.run_all:
        sessions_to_run = [
            (sub, ses)
            for sub, sessions in registry.items()
            for ses in sessions
        ]
    else:
        sessions_to_run = [(args.subject, args.session)]

    log.info("Preprocessing %d session(s).", len(sessions_to_run))
    successful, failed = [], []
    for sub, ses in sessions_to_run:
        log.info("─── %s / %s ───", sub, ses)
        try:
            preprocess_session(sub, ses)
            successful.append(f"{sub}/{ses}")
        except Exception:
            failed.append(f"{sub}/{ses}")
            traceback.print_exc()

    log.info("")
    log.info("Done: %d/%d succeeded.", len(successful), len(sessions_to_run))
    if failed:
        log.error("Failed: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
