"""Stage 02 — windowed dual LCMV beamformer.

Reads the session_data.npz produced by 01_preprocess.py, applies CAR and
bandpass filtering, builds steering dictionaries, clusters organ surfaces via
spherical k-means, runs the windowed dual LCMV beamformer, and performs leakage
subtraction.  Results are saved to data/derivatives/analysis/.

Usage
-----
# Single session
python scripts/02_lcmv.py --subject sub-06 --session ses-20250828

# All registered sessions
python scripts/02_lcmv.py --all
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── Project root on sys.path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mclcmv.beamforming.clustering import build_cluster_representatives
from mclcmv.beamforming.lcmv import leakage_subtraction, run_windowed_lcmv
from mclcmv.config import (
    BLADDER_BAND_HZ,
    DERIV_ANALYSIS,
    DERIV_SYNCED,
    DIAG_LOAD_FACTOR,
    MAX_SURFACE_VERTS,
    N_CLUSTERS,
    N_KMEANS_ITER,
    N_NULL_DIRECTIONS,
    SHRINKAGE_BETA,
    STEERING_EXPONENT,
    STEP_SIZE_SEC,
    UTERUS_BAND_HZ,
    WINDOW_SIZE_SEC,
    load_active_sessions,
    load_orientation_registry,
)
from mclcmv.forward.steering import build_steering_dictionary, compute_doa_vectors
from mclcmv.preprocessing.ehg import (
    apply_bandpass,
    apply_car,
    design_bandpass,
    detect_bad_channels,
    rational_resample,
)

TARGET_FS: float = 500.0   # Hz — all sessions normalised to this rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lcmv")


def run_lcmv_session(subject_id: str, session_id: str) -> Path:
    """Run the LCMV pipeline for one session.

    Reads  : data/derivatives/synced/<subject>/<session>/session_data.npz
    Writes : data/derivatives/analysis/<subject>/<session>/{config.json,
                                                             results_light.npz,
                                                             results_full.pkl}

    For sessions without a uterus segmentation (e.g. male subjects) the
    pipeline runs in **bladder-only mode**: only the bladder beamformer is
    computed and ``y_uterus`` is an all-zeros array.

    Returns
    -------
    Path to the output directory.
    """
    t0 = time.time()

    synced_dir = DERIV_SYNCED / subject_id / session_id
    npz_path   = synced_dir / "session_data.npz"
    meta_path  = synced_dir / "meta.json"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"session_data.npz not found at {npz_path}. "
            "Run 01_preprocess.py first."
        )

    # ── Load preprocessed data ─────────────────────────────────────────────
    data = np.load(npz_path, allow_pickle=False)
    ehg_data        = data["ehg_data"]          # (n_ch, n_samples) trimmed+clipped
    fs              = float(data["sampling_rate"])
    electrode_coords= data["electrode_coords"]  # (n_elec, 3) cm
    uterus_verts    = data["uterus_verts"]       # (G_U, 3) cm
    bladder_verts   = data["bladder_verts"]      # (G_B, 3) cm

    n_ch, n_samples = ehg_data.shape
    log.info("Loaded: %d ch × %d samples @ %.0f Hz  (%.1f min)",
             n_ch, n_samples, fs, n_samples / fs / 60)

    with open(meta_path) as fh:
        meta = json.load(fh)

    has_uterus = len(uterus_verts) > 0
    if not has_uterus:
        log.info(
            "No uterus surface for %s/%s (sex=%s) — running in bladder-only mode.",
            subject_id, session_id, meta.get("sex", "unknown"),
        )

    # ── Resample to TARGET_FS if needed ───────────────────────────────────
    if abs(fs - TARGET_FS) > 1.0:
        log.info("Resampling %.0f Hz → %.0f Hz …", fs, TARGET_FS)
        ehg_data, fs, (up, down) = rational_resample(ehg_data, fs, TARGET_FS)
        n_samples = ehg_data.shape[1]
        log.info("  After resampling: %d samples @ %.0f Hz  (up=%d, down=%d)",
                 n_samples, fs, up, down)

    # ── EHG preprocessing ─────────────────────────────────────────────────
    log.info("Bad channels + CAR …")
    good_ch = detect_bad_channels(ehg_data)
    n_bad   = int((~good_ch).sum())
    if n_bad:
        log.warning("  %d bad channel(s): %s", n_bad, np.where(~good_ch)[0].tolist())
    data_car = apply_car(ehg_data, good_channels=good_ch)

    log.info("Bandpass filtering (uterus + bladder bands) …")
    sos_ut = design_bandpass(fs, low=UTERUS_BAND_HZ[0],  high=UTERUS_BAND_HZ[1])
    sos_bl = design_bandpass(fs, low=BLADDER_BAND_HZ[0], high=BLADDER_BAND_HZ[1])
    # padlen = 100 s of reflection padding — mitigates edge transients from
    # the short (~7 min) trimmed recordings at the very low filter cutoff.
    padlen = int(100 * fs)
    data_ut = apply_bandpass(data_car, sos_ut, padlen=padlen)
    data_bl = apply_bandpass(data_car, sos_bl, padlen=padlen)

    # ── Steering dictionaries ──────────────────────────────────────────────
    log.info("Building steering dictionaries (p=%.1f) …", STEERING_EXPONENT)
    array_center = electrode_coords.mean(axis=0)
    D_bl = build_steering_dictionary(electrode_coords, bladder_verts, p=STEERING_EXPONENT)
    if has_uterus:
        D_ut = build_steering_dictionary(electrode_coords, uterus_verts, p=STEERING_EXPONENT)
        log.info("  D_ut: %s,  D_bl: %s", D_ut.shape, D_bl.shape)
    else:
        D_ut = np.zeros((n_ch, 0))
        log.info("  D_ut: n/a (bladder-only),  D_bl: %s", D_bl.shape)

    # ── DOA vectors + spherical k-means ───────────────────────────────────
    log.info("DOA vectors + spherical k-means (K=%d) …", N_CLUSTERS)
    doa_bl = compute_doa_vectors(bladder_verts, array_center)
    S_bl, lab_bl, _ = build_cluster_representatives(
        doa_bl, D_bl, k=N_CLUSTERS, n_iter=N_KMEANS_ITER, seed=1)
    if has_uterus:
        doa_ut = compute_doa_vectors(uterus_verts, array_center)
        S_ut, lab_ut, _ = build_cluster_representatives(
            doa_ut, D_ut, k=N_CLUSTERS, n_iter=N_KMEANS_ITER, seed=0)
        log.info("  S_ut: %s,  S_bl: %s", S_ut.shape, S_bl.shape)
    else:
        doa_ut  = np.zeros((0, 3))
        S_ut    = None                               # triggers bladder-only mode
        lab_ut  = np.zeros(0, dtype=np.intp)
        log.info("  S_ut: n/a (bladder-only),  S_bl: %s", S_bl.shape)

    # ── Windowed dual LCMV ─────────────────────────────────────────────────
    log.info(
        "Windowed LCMV  (window=%.0f s, step=%.0f s, β=%.2f, "
        "diag_load=%.0e, n_null=%d) …",
        WINDOW_SIZE_SEC, STEP_SIZE_SEC, SHRINKAGE_BETA,
        DIAG_LOAD_FACTOR, N_NULL_DIRECTIONS,
    )
    y_uterus, y_bladder, diagnostics = run_windowed_lcmv(
        data_uterus_band  = data_ut,
        data_bladder_band = data_bl,
        S_uterus          = S_ut,
        S_bladder         = S_bl,
        fs                = fs,
        window_size_sec   = WINDOW_SIZE_SEC,
        step_size_sec     = STEP_SIZE_SEC,
        beta              = SHRINKAGE_BETA,
        diag_load         = DIAG_LOAD_FACTOR,
        n_null            = N_NULL_DIRECTIONS,
    )
    n_windows = len(diagnostics)
    log.info("  %d windows processed.", n_windows)

    if diagnostics:
        bl_pass = float(np.mean([d["bl_pass_residual"] for d in diagnostics]))
        bl_null = float(np.mean([d["bl_null_residual"] for d in diagnostics]))
        if has_uterus:
            ut_pass = float(np.mean([d["ut_pass_residual"] for d in diagnostics]))
            ut_null = float(np.mean([d["ut_null_residual"] for d in diagnostics]))
            log.info(
                "  Constraint residuals (mean) — "
                "ut_pass=%.2e  ut_null=%.2e  bl_pass=%.2e  bl_null=%.2e",
                ut_pass, ut_null, bl_pass, bl_null,
            )
        else:
            ut_pass = ut_null = None
            log.info(
                "  Constraint residuals (mean, bladder-only) — "
                "bl_pass=%.2e  bl_null=%.2e",
                bl_pass, bl_null,
            )

    # ── Leakage subtraction ────────────────────────────────────────────────
    # Use the median of per-window leakage coefficients as the global α.
    # This is more robust than full-signal OLS: the WOLA signals can have
    # global drift that inflates the OLS estimate, while per-window α
    # reflects the true short-term leakage seen by the beamformer.
    alpha_per_window = [d["alpha_leakage"] for d in diagnostics
                        if np.isfinite(d.get("alpha_leakage", np.nan))]
    if alpha_per_window:
        alpha = float(np.median(alpha_per_window))
    else:
        _, alpha = leakage_subtraction(y_uterus, y_bladder)
    y_bladder_clean = y_bladder - alpha * y_uterus
    if has_uterus:
        log.info("Leakage subtraction …  α = %.4f  (median of %d per-window values)",
                 alpha, len(alpha_per_window))
    else:
        log.info("Bladder-only mode — leakage subtraction not applicable (α = 0).")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = DERIV_ANALYSIS / subject_id / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving to %s …", out_dir)

    config_out = {
        "meta": {
            "subject_id":   subject_id,
            "session_id":   session_id,
            "sex":          meta.get("sex"),
            "cycle_phase":  meta.get("cycle_phase"),
            "bladder_only": not has_uterus,
        },
        "config": {
            "fs":                  float(fs),
            "window_size_samples": int(WINDOW_SIZE_SEC * fs),
            "step_size_samples":   int(STEP_SIZE_SEC   * fs),
            "window_size_sec":     WINDOW_SIZE_SEC,
            "step_size_sec":       STEP_SIZE_SEC,
            "p":                   STEERING_EXPONENT,
            "beta":                SHRINKAGE_BETA,
            "diag_load":           DIAG_LOAD_FACTOR,
            "K_ut":                N_CLUSTERS,
            "K_bl":                N_CLUSTERS,
            "pass_mode":           "mean",
            "n_null":              N_NULL_DIRECTIONS,
            "uterus_band_hz":      list(UTERUS_BAND_HZ),
            "bladder_band_hz":     list(BLADDER_BAND_HZ),
            "max_surface_verts":   MAX_SURFACE_VERTS,
        },
        "results_summary": {
            "n_channels":        n_ch,
            "n_mri_samples":     int(n_samples),
            "n_windows":         n_windows,
            "alpha_leakage":     float(alpha),
            "mean_ut_pass_res":  ut_pass if diagnostics else None,
            "mean_ut_null_res":  ut_null if diagnostics else None,
            "mean_bl_pass_res":  float(bl_pass) if diagnostics else None,
            "mean_bl_null_res":  float(bl_null) if diagnostics else None,
        },
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config_out, fh, indent=2)

    t_axis = np.arange(n_samples) / fs
    S_ut_save = S_ut if has_uterus else np.zeros((n_ch, 0), dtype=complex)
    np.savez_compressed(
        out_dir / "results_light.npz",
        t               = t_axis,
        uterus_ts       = y_uterus,
        bladder_ts      = y_bladder,
        bladder_clean_ts= y_bladder_clean,
        lab_ut          = lab_ut,
        lab_bl          = lab_bl,
        uterus_doa      = doa_ut,
        bladder_doa     = doa_bl,
        S_ut_clusters   = S_ut_save,
        S_bl_clusters   = S_bl,
    )

    # Extract per-window weight vectors for null-gain computation in analysis
    w_ut_list = [d.pop("w_ut") for d in diagnostics]
    w_bl_list = [d.pop("w_bl") for d in diagnostics]

    results_full = {
        "meta":             config_out["meta"],
        "config":           config_out["config"],
        "sensor_positions": electrode_coords,
        "uterus_doa":       doa_ut,
        "bladder_doa":      doa_bl,
        "lab_ut":           lab_ut,
        "lab_bl":           lab_bl,
        "S_ut_clusters":    S_ut_save,
        "S_bl_clusters":    S_bl,
        "t":                t_axis,
        "uterus_ts":        y_uterus,
        "bladder_ts":       y_bladder,
        "bladder_clean_ts": y_bladder_clean,
        "alpha":            float(alpha),
        "diagnostics":      diagnostics,
        "w_ut":             w_ut_list,   # list of (M,) arrays, one per window
        "w_bl":             w_bl_list,   # list of (M,) arrays, one per window
    }
    with open(out_dir / "results_full.pkl", "wb") as fh:
        pickle.dump(results_full, fh, protocol=4)

    log.info("Session done in %.1f s  ✓", time.time() - t0)
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 02: run LCMV beamformer for one or all sessions."
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

    if args.run_all:
        sessions_to_run = load_active_sessions()
    else:
        sessions_to_run = [(args.subject, args.session)]

    log.info("Running LCMV for %d session(s).", len(sessions_to_run))
    successful, failed = [], []
    for sub, ses in sessions_to_run:
        log.info("═══ %s / %s ═══", sub, ses)
        try:
            run_lcmv_session(sub, ses)
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
