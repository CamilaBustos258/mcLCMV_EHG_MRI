"""Stage 00 — Synthetic demo for the mcLCMV pipeline.

Generates synthetic but anatomically plausible EHG data from two known sources
(uterus and bladder), runs the complete windowed LCMV pipeline, and saves 7
diagnostic figures with ground-truth separation verification.

No real participant data required.

Design notes
------------
Two issues arise when testing LCMV with synthetic data at physiological EHG
frequencies (0.005–0.05 Hz):

1. Narrow-band noise from a 480 s record has only ~22 effective degrees of
   freedom, so two independent realizations are highly correlated (r > 0.98).
   Fix: use non-overlapping demo bands (0.1–0.5 Hz and 0.5–2.0 Hz) that give
   ~192 and ~720 DoF respectively, ensuring r(s_U, s_B) < 0.01.

2. The paper uses K=3 cluster representatives as null columns, but all three
   cluster reps point in nearly the same direction (cos angle > 0.99), so
   2 of 3 reps leave a residual component of the opposing-organ mean vector
   outside the null subspace.  For the synthetic ground-truth test we instead
   use the top-2 SVD directions of the full steering dictionary, which span
   99.999 % of the opposing organ's mean direction and make the null
   effectively exact.  The paper's cluster-rep null is the production default;
   the SVD null is used here purely to give a clean ground-truth verification.

Usage
-----
    python scripts/00_demo.py

Outputs
-------
    data/demo/figures/01_geometry.png
    data/demo/figures/02_steering_heatmaps.png
    data/demo/figures/03_beamformer_response.png
    data/demo/figures/04_time_series.png
    data/demo/figures/05_constraint_residuals.png
    data/demo/figures/06_leakage_per_window.png
    data/demo/figures/07_separation_quality.png
    data/demo/metrics.txt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mclcmv.beamforming.clustering import build_cluster_representatives
from mclcmv.beamforming.lcmv import (
    add_diagonal_loading,
    constraint_residual,
    hann_window,
    lcmv_weights,
    sample_covariance,
    shrink_covariance,
)
from mclcmv.config import (
    DIAG_LOAD_FACTOR,
    N_CLUSTERS,
    N_KMEANS_ITER,
    N_NULL_DIRECTIONS,
    SHRINKAGE_BETA,
    STEERING_EXPONENT,
    STEP_SIZE_SEC,
    WINDOW_SIZE_SEC,
)
from mclcmv.forward.steering import build_steering_dictionary, compute_doa_vectors

# ── Run parameters ─────────────────────────────────────────────────────────
FS       = 500.0
DURATION = 8 * 60          # seconds
N        = int(DURATION * FS)
# Higher SNR than real EHG (typically 5–10 dB) so that the ground-truth
# correlations are clearly visible despite the finite null-subspace coverage.
SNR_DB   = 25.0
SEED     = 42
DSAMP    = 50              # display decimation for time-series plots

# Demo source bands — non-overlapping so the two synthetic sources are
# statistically independent (see design notes in module docstring).
DEMO_UT_BAND = (0.1, 0.5)    # Hz  — slow uterine-contraction-like oscillations
DEMO_BL_BAND = (0.5, 2.0)    # Hz  — slightly faster bladder-activity-like signal

OUT_DIR     = PROJECT_ROOT / "data" / "demo" / "figures"
METRICS_OUT = PROJECT_ROOT / "data" / "demo" / "metrics.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("demo")
RNG = np.random.default_rng(SEED)


# ══ Geometry helpers ════════════════════════════════════════════════════════

def make_electrode_array() -> np.ndarray:
    """Return (8, 3) electrode positions: 2-row × 4-col grid, 2 cm spacing."""
    return np.array(
        [[col * 2.0, row * 3.0, 0.0] for row in range(2) for col in range(4)],
        dtype=float,
    )


def make_ellipsoid_surface(
    center: np.ndarray, a: float, b: float, c: float,
    n_theta: int = 20, n_phi: int = 30,
) -> np.ndarray:
    """(G, 3) vertices on an ellipsoid with semi-axes a, b, c."""
    theta = np.linspace(0.1, np.pi - 0.1, n_theta)
    phi   = np.linspace(0.0, 2 * np.pi,   n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    verts = np.column_stack([
        (center[0] + a * np.sin(TH) * np.cos(PH)).ravel(),
        (center[1] + b * np.sin(TH) * np.sin(PH)).ravel(),
        (center[2] + c * np.cos(TH)).ravel(),
    ])
    return verts


def make_sphere_surface(
    center: np.ndarray, radius: float,
    n_theta: int = 15, n_phi: int = 20,
) -> np.ndarray:
    """(G, 3) vertices on a sphere."""
    return make_ellipsoid_surface(center, radius, radius, radius, n_theta, n_phi)


# ══ Signal helpers ══════════════════════════════════════════════════════════

def bandlimited_noise(
    n: int, low: float, high: float, rng: np.random.Generator,
) -> np.ndarray:
    """Unit-variance bandlimited noise in [low, high] Hz."""
    white = rng.standard_normal(n)
    sos   = butter(2, [low, high], btype="bandpass", fs=FS, output="sos")
    sig   = sosfiltfilt(sos, white)
    return sig / (sig.std() + 1e-12)


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-subtracted Pearson correlation."""
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ══ Figures ══════════════════════════════════════════════════════════════════

def fig01_geometry(
    electrode_coords: np.ndarray,
    uterus_verts: np.ndarray,
    bladder_verts: np.ndarray,
    doa_ut: np.ndarray,
    doa_bl: np.ndarray,
    lab_ut: np.ndarray,
    lab_bl: np.ndarray,
    array_center: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Organ surfaces (subsampled for speed)
    step = max(1, len(uterus_verts) // 300)
    ax.scatter(*uterus_verts[::step].T,  s=4,  c="salmon",     alpha=0.3, label="Uterus surface")
    step = max(1, len(bladder_verts) // 300)
    ax.scatter(*bladder_verts[::step].T, s=4,  c="dodgerblue", alpha=0.3, label="Bladder surface")

    # Electrodes
    ax.scatter(*electrode_coords.T, s=120, c="black", marker="^",
               zorder=5, label="Electrodes")

    # Cluster centroids
    colors_ut = ["firebrick", "tomato", "orangered"]
    colors_bl = ["navy",      "royalblue", "steelblue"]
    for k in range(N_CLUSTERS):
        c_ut = doa_ut[lab_ut == k].mean(axis=0)
        c_ut /= np.linalg.norm(c_ut) + 1e-12
        pt   = array_center + c_ut * 6
        ax.scatter(*pt, s=120, c=colors_ut[k], marker="D",
                   label=f"UT cluster {k}" if k == 0 else f"UT {k}")
        c_bl = doa_bl[lab_bl == k].mean(axis=0)
        c_bl /= np.linalg.norm(c_bl) + 1e-12
        pt   = array_center + c_bl * 6
        ax.scatter(*pt, s=120, c=colors_bl[k], marker="s",
                   label=f"BL cluster {k}" if k == 0 else f"BL {k}")

    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
    ax.set_title("Synthetic geometry: electrode array + organ surfaces + cluster centroids")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    plt.tight_layout()
    path = OUT_DIR / "01_geometry.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


def fig02_steering_heatmaps(S_ut: np.ndarray, S_bl: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, S, title, cmap in zip(
        axes,
        [np.abs(S_ut), np.abs(S_bl)],
        ["Uterus cluster steering vectors", "Bladder cluster steering vectors"],
        ["Reds", "Blues"],
    ):
        im = ax.imshow(S, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Electrode index")
        ax.set_xticks(range(S.shape[1]))
        ax.set_xticklabels([f"C{k}" for k in range(S.shape[1])])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="|steering|")
    plt.tight_layout()
    path = OUT_DIR / "02_steering_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


def fig03_beamformer_response(
    w_ut: np.ndarray, w_bl: np.ndarray,
    S_ut: np.ndarray, S_bl: np.ndarray,
) -> None:
    """Gain in dB for each cluster, for each beamformer."""
    eps   = 1e-12
    S_all = np.hstack([S_ut, S_bl])   # (M, 2K)
    labels = ([f"UT{k}" for k in range(S_ut.shape[1])] +
              [f"BL{k}" for k in range(S_bl.shape[1])])
    x     = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, w, bfname in zip(axes, [w_ut, w_bl], ["Uterus BF  (w_ut)", "Bladder BF  (w_bl)"]):
        gains_db = [20 * np.log10(abs(w.conj() @ S_all[:, i]) + eps)
                    for i in range(S_all.shape[1])]
        colors = (["firebrick"] * S_ut.shape[1] + ["navy"] * S_bl.shape[1])
        bars = ax.bar(x, gains_db, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", label="0 dB (unity gain)")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Gain (dB)"); ax.set_title(bfname)
        ax.legend(fontsize=8)
        for i, (bar, g) in enumerate(zip(bars, gains_db)):
            ax.text(bar.get_x() + bar.get_width() / 2, g + 0.5,
                    f"{g:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Beamformer cluster response — UT clusters ≈0 dB for own BF, "
                 "BL clusters suppressed", fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "03_beamformer_response.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


def fig04_time_series(
    s_U: np.ndarray, s_B: np.ndarray,
    y_uterus: np.ndarray, y_bladder_clean: np.ndarray,
    r_ut_sU: float, r_bl_sB: float,
    trim: int,
) -> None:
    # Only plot the interior (trimmed) region
    t_min = np.arange(trim, N - trim)[::DSAMP] / FS / 60   # time axis in minutes

    def _ds(sig: np.ndarray, t0: int, t1: int) -> np.ndarray:
        s = sig[t0:t1][::DSAMP]
        return s / (np.std(s) + 1e-12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    pairs = [
        (s_U,           "True source  $s_U(t)$",                                "dimgray"),
        (y_uterus,      f"Reconstructed  $\\hat{{y}}_U$   r={r_ut_sU:+.2f}",   "firebrick"),
        (s_B,           "True source  $s_B(t)$",                                "dimgray"),
        (y_bladder_clean, f"Reconstructed  $\\hat{{y}}_B$ (clean)   r={r_bl_sB:+.2f}", "navy"),
    ]
    for ax, (sig, lbl, col) in zip(axes, pairs):
        ax.plot(t_min, _ds(sig, trim, N - trim), color=col, linewidth=0.7)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_ylim(-4, 4)
        ax.axhline(0, color="k", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (min)  [interior region shown]")
    fig.suptitle("Time series: ground-truth sources vs. LCMV reconstructions", fontsize=10)
    plt.tight_layout()
    path = OUT_DIR / "04_time_series.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


def fig05_constraint_residuals(diagnostics: list[dict]) -> None:
    idx      = [d["window_idx"] for d in diagnostics]
    ut_pass  = [d["ut_pass_residual"] for d in diagnostics]
    ut_null  = [d["ut_null_residual"] for d in diagnostics]
    bl_pass  = [d["bl_pass_residual"] for d in diagnostics]
    bl_null  = [d["bl_null_residual"] for d in diagnostics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.semilogy(idx, ut_pass, "o-", color="firebrick", label="UT pass", markersize=4)
    ax1.semilogy(idx, bl_pass, "s-", color="navy",      label="BL pass", markersize=4)
    ax1.set_ylabel("‖δ‖∞  pass")
    ax1.legend(fontsize=8); ax1.set_title("Pass constraint residuals  (should be ≪ 1)")

    ax2.semilogy(idx, [max(v, 1e-18) for v in ut_null], "o-", color="firebrick", label="UT null", markersize=4)
    ax2.semilogy(idx, [max(v, 1e-18) for v in bl_null], "s-", color="navy",      label="BL null", markersize=4)
    ax2.set_ylabel("‖δ‖∞  null")
    ax2.set_xlabel("Window index")
    ax2.legend(fontsize=8); ax2.set_title("Null constraint residuals  (SVD null → machine-ε)")

    plt.tight_layout()
    path = OUT_DIR / "05_constraint_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


def fig06_leakage(diagnostics: list[dict], alpha_global: float) -> None:
    idx     = [d["window_idx"] for d in diagnostics]
    alphas  = [d["alpha_leakage"] for d in diagnostics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(idx, alphas, color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.4,
           label="Per-window α_w")
    ax.axhline(alpha_global, color="firebrick", linewidth=1.5, linestyle="--",
               label=f"Median α = {alpha_global:.3f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Leakage coefficient  α_w")
    ax.set_title("Per-window leakage coefficient — near-zero indicates effective null constraint")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "06_leakage_per_window.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


def fig07_separation_quality(
    r_ut_sU: float, r_ut_sB: float,
    r_bl_sB: float, r_bl_sU: float,
    r_blc_sB: float, r_blc_sU: float,
) -> None:
    groups    = ["Uterus BF", "Bladder BF\n(before leakage)", "Bladder BF\n(after leakage)"]
    target    = [r_ut_sU,  r_bl_sB,  r_blc_sB]
    crosstalk = [r_ut_sB,  r_bl_sU,  r_blc_sU]

    x     = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, target,    width, label="Target organ correlation",
                color=["firebrick", "navy", "navy"], alpha=0.8, edgecolor="k")
    b2 = ax.bar(x + width / 2, crosstalk, width, label="Cross-organ correlation",
                color=["salmon",    "steelblue", "steelblue"], alpha=0.8, edgecolor="k")

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Pearson r  with ground-truth source")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title("Ground-truth separation quality\n"
                 "Target correlation → high   |   Cross-organ correlation → near 0")
    ax.legend(fontsize=9)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.01 if h >= 0 else h - 0.03,
                f"{h:+.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "07_separation_quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path.name)


# ══ Main ════════════════════════════════════════════════════════════════════

def main() -> None:
    records: list[str] = []

    def record(msg: str) -> None:
        log.info(msg)
        records.append(msg)

    # ── 1. Geometry ────────────────────────────────────────────────────────
    log.info("═══ 1 / 7  Synthetic geometry ═══")
    electrode_coords = make_electrode_array()                       # (8, 3) cm
    array_center     = electrode_coords.mean(axis=0)               # (3,)

    # Uterus: ellipsoid, centred directly behind the array at 9 cm depth
    uterus_verts  = make_ellipsoid_surface(
        np.array([3.0, 1.5, 9.0]), a=3.0, b=2.0, c=2.5)

    # Bladder: sphere, more anterior (y+3.5) and shallower (z=7 cm)
    bladder_verts = make_sphere_surface(
        np.array([3.0, 5.0, 7.0]), radius=2.5)

    log.info("  Electrodes: %d  |  uterus verts: %d  |  bladder verts: %d",
             len(electrode_coords), len(uterus_verts), len(bladder_verts))

    # ── 2. Steering dicts + clustering ────────────────────────────────────
    log.info("═══ 2 / 7  Steering dictionaries + clustering ═══")
    D_ut = build_steering_dictionary(electrode_coords, uterus_verts,  p=STEERING_EXPONENT)
    D_bl = build_steering_dictionary(electrode_coords, bladder_verts, p=STEERING_EXPONENT)

    doa_ut = compute_doa_vectors(uterus_verts,  array_center)
    doa_bl = compute_doa_vectors(bladder_verts, array_center)

    S_ut, lab_ut, _ = build_cluster_representatives(
        doa_ut, D_ut, k=N_CLUSTERS, n_iter=N_KMEANS_ITER, seed=0)
    S_bl, lab_bl, _ = build_cluster_representatives(
        doa_bl, D_bl, k=N_CLUSTERS, n_iter=N_KMEANS_ITER, seed=1)

    # Separation angle between cluster centroids
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    sep_angles = []
    for ku in range(N_CLUSTERS):
        cu = _unit(doa_ut[lab_ut == ku].mean(axis=0))
        for kb in range(N_CLUSTERS):
            cb = _unit(doa_bl[lab_bl == kb].mean(axis=0))
            sep_angles.append(np.degrees(np.arccos(np.clip(np.dot(cu, cb), -1, 1))))
    sep_mean = float(np.mean(sep_angles))
    record(f"  Separation angle: {sep_mean:.1f}° (mean of {N_CLUSTERS**2} pairs)")

    # ── 3. SVD-based null columns ─────────────────────────────────────────
    # For the ground-truth test the null columns are the top-N_NULL_DIRECTIONS
    # SVD directions of the full opposing-organ steering dictionary.  This
    # spans 99.999% of that organ's mean steering vector (vs. ~99.9% with 2
    # cluster reps), making the null constraint effectively exact.
    log.info("═══ 3 / 7  SVD-based null columns ═══")

    U_ut, s_ut, _ = np.linalg.svd(D_ut.real, full_matrices=False)
    U_bl, s_bl, _ = np.linalg.svd(D_bl.real, full_matrices=False)

    # Pass vectors: normalised mean of cluster representatives (same as pipeline)
    s_pass_ut = S_ut.mean(axis=1).real
    s_pass_ut /= (np.linalg.norm(s_pass_ut) + 1e-12)
    s_pass_bl = S_bl.mean(axis=1).real
    s_pass_bl /= (np.linalg.norm(s_pass_bl) + 1e-12)

    a_U = s_pass_ut.copy()   # ground-truth source direction for uterus
    a_B = s_pass_bl.copy()   # ground-truth source direction for bladder

    # Null columns: top-N_NULL_DIRECTIONS SVD directions of opposing organ dict
    Q_null_ut = U_bl[:, :N_NULL_DIRECTIONS].astype(complex)   # for uterus BF
    Q_null_bl = U_ut[:, :N_NULL_DIRECTIONS].astype(complex)   # for bladder BF

    # Constraint matrices: C = [s_pass | Q_null], f = [1, 0, …, 0]^T
    L = 1 + N_NULL_DIRECTIONS
    f_vec = np.zeros(L, dtype=complex); f_vec[0] = 1.0
    C_ut = np.hstack([s_pass_ut[:, None].astype(complex), Q_null_ut])  # (M, L)
    C_bl = np.hstack([s_pass_bl[:, None].astype(complex), Q_null_bl])  # (M, L)

    # Verify null coverage of opposing-organ mean direction
    proj_bl = Q_null_ut.real @ (Q_null_ut.real.T @ a_B)
    proj_ut = Q_null_bl.real @ (Q_null_bl.real.T @ a_U)
    record(f"  SVD null coverage of a_B in ut-null subspace: "
           f"{np.sum(proj_bl**2):.6f}  (1.000 = perfect)")
    record(f"  SVD null coverage of a_U in bl-null subspace: "
           f"{np.sum(proj_ut**2):.6f}  (1.000 = perfect)")

    # ── 4. Generating synthetic EHG ───────────────────────────────────────
    log.info("═══ 4 / 7  Generating synthetic EHG (%d s, %.0f Hz) ═══",
             DURATION, FS)
    # Non-overlapping frequency bands → statistically independent sources.
    # Real EHG uses 0.005–0.05 Hz, but those bands have only ~22 effective
    # degrees of freedom in 8 min, making two realisations highly correlated
    # (r > 0.98).  Demo bands give ~192 DoF (uterus) and ~720 DoF (bladder).
    s_U = bandlimited_noise(N, *DEMO_UT_BAND, RNG)
    s_B = bandlimited_noise(N, *DEMO_BL_BAND, RNG)
    r_sources = pearsonr(s_U, s_B)
    record(f"  r(s_U, s_B) = {r_sources:+.4f}  (near zero = independent sources)")

    sig_power  = float(np.mean((np.outer(a_U, s_U)) ** 2))
    noise_std  = np.sqrt(sig_power / (10 ** (SNR_DB / 10)))
    ehg_data   = (np.outer(a_U, s_U)
                + np.outer(a_B, s_B)
                + RNG.standard_normal((8, N)) * noise_std)   # (8, N)
    record(f"  SNR = {SNR_DB:.1f} dB  |  noise_std = {noise_std:.5f}")

    # ── 5. EHG preprocessing ──────────────────────────────────────────────
    log.info("═══ 5 / 7  EHG preprocessing ═══")
    # Bandpass / CAR are skipped for the synthetic demo.  Sources were
    # generated as bandlimited noise so the signal is already band-limited,
    # and the CAR would modify the effective steering vectors.
    record("  Bandpass/CAR skipped (sources already band-limited by construction)")

    # ── 6. Windowed LCMV with SVD-based constraints ───────────────────────
    log.info("═══ 6 / 7  Windowed LCMV ═══")
    win_len  = int(WINDOW_SIZE_SEC * FS)
    step     = int(STEP_SIZE_SEC * FS)
    n_windows = max(0, (N - win_len) // step + 1)

    hann      = hann_window(win_len)
    y_ut_acc  = np.zeros(N)
    y_bl_acc  = np.zeros(N)
    norm_acc  = np.zeros(N)
    diagnostics: list[dict] = []

    from tqdm import tqdm  # local import for cleaner top-level imports
    for wi in tqdm(range(n_windows), desc="LCMV windows", leave=False):
        start = wi * step
        end   = start + win_len
        seg   = ehg_data[:, start:end].astype(complex)

        R = sample_covariance(seg)
        R = shrink_covariance(R, beta=SHRINKAGE_BETA)
        R = add_diagonal_loading(R, load_factor=DIAG_LOAD_FACTOR)

        w_ut = lcmv_weights(R, C_ut, f_vec)
        w_bl = lcmv_weights(R, C_bl, f_vec)

        ut_pass_res = constraint_residual(C_ut[:, :1], w_ut, f_vec[:1])
        ut_null_res = float(np.max(np.abs(C_ut[:, 1:].conj().T @ w_ut)))
        bl_pass_res = constraint_residual(C_bl[:, :1], w_bl, f_vec[:1])
        bl_null_res = float(np.max(np.abs(C_bl[:, 1:].conj().T @ w_bl)))

        y_ut_w = (w_ut.conj() @ seg).real
        y_bl_w = (w_bl.conj() @ seg).real

        denom_w = float(np.dot(y_ut_w, y_ut_w))
        alpha_w = (float(np.dot(y_bl_w, y_ut_w)) / denom_w
                   if denom_w > 1e-12 else 0.0)

        diagnostics.append({
            "window_idx":       wi,
            "t0":               int(start),
            "t1":               int(end),
            "ut_pass_residual": ut_pass_res,
            "ut_null_residual": ut_null_res,
            "bl_pass_residual": bl_pass_res,
            "bl_null_residual": bl_null_res,
            "alpha_leakage":    alpha_w,
            "w_ut":             w_ut.copy(),
            "w_bl":             w_bl.copy(),
        })

        y_ut_acc[start:end] += y_ut_w * hann
        y_bl_acc[start:end] += y_bl_w * hann
        norm_acc[start:end] += hann ** 2

    # WOLA normalisation
    safe_norm = np.where(norm_acc < 1e-12, 1.0, norm_acc)
    y_uterus  = y_ut_acc / safe_norm
    y_bladder = y_bl_acc / safe_norm

    record(f"  Windows processed: {len(diagnostics)}")

    # ── 7. Leakage subtraction ────────────────────────────────────────────
    alpha_vals = [d["alpha_leakage"] for d in diagnostics
                  if np.isfinite(d.get("alpha_leakage", np.nan))]
    alpha           = float(np.median(alpha_vals)) if alpha_vals else 0.0
    y_bladder_clean = y_bladder - alpha * y_uterus
    record(f"  Leakage coefficient α (median) = {alpha:.4f}")

    # ── 8. Ground-truth separation metrics ────────────────────────────────
    # Correlations are computed on the interior region [win_len : N-win_len],
    # where the WOLA Hann normalization is well-conditioned (≥ 4 overlapping
    # windows).  Edge samples where fewer windows overlap can have large
    # WOLA-normalisation artefacts and are excluded.
    log.info("═══ 7 / 7  Ground-truth separation metrics ═══")
    trim = win_len
    sl   = slice(trim, N - trim)

    r_ut_sU  = pearsonr(y_uterus[sl],       s_U[sl])
    r_ut_sB  = pearsonr(y_uterus[sl],       s_B[sl])
    r_bl_sB  = pearsonr(y_bladder[sl],      s_B[sl])
    r_bl_sU  = pearsonr(y_bladder[sl],      s_U[sl])
    r_blc_sB = pearsonr(y_bladder_clean[sl], s_B[sl])
    r_blc_sU = pearsonr(y_bladder_clean[sl], s_U[sl])

    record("")
    record("── Ground-truth correlations (interior region) ────────────────")
    record(f"  r(y_ut,       s_U) = {r_ut_sU:+.3f}   ← high: uterus BF recovers uterus")
    record(f"  r(y_ut,       s_B) = {r_ut_sB:+.3f}   ← ~0: bladder nulled in uterus BF")
    record(f"  r(y_bl,       s_B) = {r_bl_sB:+.3f}   ← high: bladder BF recovers bladder")
    record(f"  r(y_bl,       s_U) = {r_bl_sU:+.3f}   ← ~0: uterus nulled in bladder BF")
    record(f"  r(y_bl_clean, s_U) = {r_blc_sU:+.3f}   ← ~0: leakage removed")
    record(f"  r(y_bl_clean, s_B) = {r_blc_sB:+.3f}   ← stable: bladder signal preserved")

    # Constraint residuals summary
    ut_pass = float(np.mean([d["ut_pass_residual"] for d in diagnostics]))
    ut_null = float(np.mean([d["ut_null_residual"] for d in diagnostics]))
    bl_pass = float(np.mean([d["bl_pass_residual"] for d in diagnostics]))
    bl_null = float(np.mean([d["bl_null_residual"] for d in diagnostics]))
    record("")
    record("── Constraint residuals (mean ‖δ‖∞) ──────────────────────────")
    record(f"  ut_pass = {ut_pass:.2e}  (expected ≪ 1)")
    record(f"  ut_null = {ut_null:.2e}  (SVD null → machine-ε)")
    record(f"  bl_pass = {bl_pass:.2e}  (expected ≪ 1)")
    record(f"  bl_null = {bl_null:.2e}  (SVD null → machine-ε)")

    for line in records:
        print(line)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text("\n".join(records))
    log.info("Metrics saved → %s", METRICS_OUT)

    # ── 9. Figures ────────────────────────────────────────────────────────
    log.info("═══ Saving figures → %s ═══", OUT_DIR)

    mid       = len(diagnostics) // 2
    w_ut_rep  = diagnostics[mid]["w_ut"]
    w_bl_rep  = diagnostics[mid]["w_bl"]

    fig01_geometry(electrode_coords, uterus_verts, bladder_verts,
                   doa_ut, doa_bl, lab_ut, lab_bl, array_center)
    fig02_steering_heatmaps(S_ut, S_bl)
    fig03_beamformer_response(w_ut_rep, w_bl_rep, S_ut, S_bl)
    fig04_time_series(s_U, s_B, y_uterus, y_bladder_clean, r_ut_sU, r_blc_sB, trim)
    fig05_constraint_residuals(diagnostics)
    fig06_leakage(diagnostics, alpha)
    fig07_separation_quality(r_ut_sU, r_ut_sB, r_bl_sB, r_bl_sU, r_blc_sB, r_blc_sU)

    log.info("Done.  All figures saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
