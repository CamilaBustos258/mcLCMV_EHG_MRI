"""Stage 03 — Compute and print the results table.

Scans data/derivatives/analysis/ for all processed sessions and computes
7 metrics for each:

    Separ.(°)  — mean±std of all K_U × K_B pairwise DOA cluster separation angles
    UT Comp.   — mean±std of Euclidean compactness across K_U uterus clusters
    BL Comp.   — mean±std of Euclidean compactness across K_B bladder clusters
    Null Gain  — mean±std of max|Q_null^H w_ut| across windows
    |α|        — median per-window absolute leakage coefficient
    Δr         — median per-window correlation change after leakage subtraction
    E Rat.     — median per-window ||y_bl_clean|| / ||y_bl||

Usage
-----
python scripts/03_results_table.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# ── Project root ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mclcmv.config import DERIV_ANALYSIS


# ── Metric helpers ─────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _cluster_centroids(doa: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Return (k, 3) unit cluster centroids in DOA space."""
    centroids = np.zeros((k, 3))
    for ki in range(k):
        members = doa[labels == ki]
        if len(members) > 0:
            centroids[ki] = _unit(members.mean(axis=0))
    return centroids


def compute_separation_angles(
    doa_ut: np.ndarray,
    lab_ut: np.ndarray,
    doa_bl: np.ndarray,
    lab_bl: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """All K_U × K_B pairwise angles between cluster centroids → (mean°, std°)."""
    C_u = _cluster_centroids(doa_ut, lab_ut, k)
    C_b = _cluster_centroids(doa_bl, lab_bl, k)
    angles = []
    for cu in C_u:
        for cb in C_b:
            dot = float(np.clip(np.dot(cu, cb), -1.0, 1.0))
            angles.append(np.degrees(np.arccos(dot)))
    a = np.array(angles)
    return float(a.mean()), float(a.std())


def compute_compactness(doa: np.ndarray, labels: np.ndarray, k: int = 3) -> tuple[float, float]:
    """Euclidean compactness per cluster (mean dist to centroid) → (mean, std)."""
    comp = []
    for ki in range(k):
        members = doa[labels == ki]
        if len(members) < 2:
            comp.append(0.0)
            continue
        c = _unit(members.mean(axis=0))
        comp.append(float(np.mean(np.linalg.norm(members - c, axis=1))))
    c_arr = np.array(comp)
    return float(c_arr.mean()), float(c_arr.std())


def _q_null_from_clusters(
    S: np.ndarray,
    energy: float = 0.95,
    sigma_floor: float = 0.10,
) -> np.ndarray:
    """SVD-based null subspace of S.  Returns (M, r) matrix."""
    S = np.asarray(S, dtype=complex)
    col_norms = np.linalg.norm(S, axis=0, keepdims=True) + 1e-12
    Sn = S / col_norms
    U, s, _ = np.linalg.svd(Sn, full_matrices=False)
    if s.size == 0:
        return np.zeros((S.shape[0], 1), dtype=complex)
    cum = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-18)
    r_energy = int(np.searchsorted(cum, energy) + 1)
    r_floor = int(np.sum(s >= sigma_floor * s[0]))
    r = max(1, min(r_energy, r_floor, S.shape[0] - 1))
    return U[:, :r]


def compute_null_gain(
    S_bl: np.ndarray,
    w_ut_list: list[np.ndarray],
) -> tuple[float, float]:
    """Per-window max(|Q_null^H w_ut|) → (mean, std) across windows."""
    Q_null = _q_null_from_clusters(S_bl)
    gains = [float(np.max(np.abs(Q_null.conj().T @ w))) for w in w_ut_list]
    g = np.array(gains)
    return float(g.mean()), float(g.std())


def _corr_zero_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation after mean-subtraction."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) + 1e-15) * (np.linalg.norm(b) + 1e-15)
    return float(np.dot(a, b) / denom)


def compute_leakage_metrics(
    diagnostics: list[dict],
    bladder_ts: np.ndarray,
    bladder_clean_ts: np.ndarray,
) -> tuple[float, float, float]:
    """Median |α|, Δr, and E Rat. from per-window diagnostics."""
    alpha_series, delta_r_series, energy_series = [], [], []

    for d in diagnostics:
        t0, t1 = d.get("t0"), d.get("t1")
        alpha_w = d.get("alpha_leakage")

        if alpha_w is not None and np.isfinite(alpha_w):
            alpha_series.append(abs(alpha_w))

        if t0 is None or t1 is None:
            continue
        t0c = max(0, int(t0))
        t1c = min(len(bladder_ts), int(t1))
        if t1c - t0c < 4:
            continue

        yb   = bladder_ts[t0c:t1c]
        ybc  = bladder_clean_ts[t0c:t1c]
        yrem = yb - ybc

        energy_series.append(float(np.linalg.norm(ybc) / (np.linalg.norm(yb) + 1e-15)))
        delta_r_series.append(_corr_zero_mean(ybc, yrem) - _corr_zero_mean(yb, yrem))

    med_alpha = float(np.nanmedian(alpha_series))   if alpha_series   else float("nan")
    med_dr    = float(np.nanmedian(delta_r_series)) if delta_r_series else float("nan")
    med_er    = float(np.nanmedian(energy_series))  if energy_series  else float("nan")
    return med_alpha, med_dr, med_er


# ── Session discovery & metrics ────────────────────────────────────────────

def discover_sessions() -> list[tuple[str, str]]:
    """Return all (subject, session) pairs that have a results_full.pkl."""
    sessions = []
    if not DERIV_ANALYSIS.exists():
        return sessions
    for pkl in sorted(DERIV_ANALYSIS.glob("*/*/results_full.pkl")):
        sessions.append((pkl.parts[-3], pkl.parts[-2]))
    return sessions


def load_session(subject_id: str, session_id: str) -> dict:
    pkl_path = DERIV_ANALYSIS / subject_id / session_id / "results_full.pkl"
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def compute_all_metrics(r: dict) -> dict:
    """Compute all 7 metrics from a results dict."""
    K = r["config"]["K_ut"]

    sep_mean, sep_std = compute_separation_angles(
        r["uterus_doa"], r["lab_ut"], r["bladder_doa"], r["lab_bl"], k=K)
    ut_comp_mean, ut_comp_std = compute_compactness(r["uterus_doa"], r["lab_ut"], k=K)
    bl_comp_mean, bl_comp_std = compute_compactness(r["bladder_doa"], r["lab_bl"], k=K)

    if "w_ut" in r and r["w_ut"]:
        ng_mean, ng_std = compute_null_gain(r["S_bl_clusters"], r["w_ut"])
    else:
        ng_mean, ng_std = float("nan"), float("nan")

    med_alpha, med_dr, med_er = compute_leakage_metrics(
        r["diagnostics"], r["bladder_ts"], r["bladder_clean_ts"])

    return {
        "sep_mean": sep_mean,     "sep_std": sep_std,
        "ut_comp_mean": ut_comp_mean, "ut_comp_std": ut_comp_std,
        "bl_comp_mean": bl_comp_mean, "bl_comp_std": bl_comp_std,
        "ng_mean": ng_mean,       "ng_std": ng_std,
        "alpha": med_alpha,
        "delta_r": med_dr,
        "energy_ratio": med_er,
    }


def fmt_pm(mean: float, std: float, dec: int = 1) -> str:
    if np.isnan(mean) or np.isnan(std):
        return "  —  "
    return f"{mean:.{dec}f}±{std:.{dec}f}"


def fmt_pm3(mean: float, std: float) -> str:
    if np.isnan(mean) or np.isnan(std):
        return "  —  "
    return f"{mean:.3f}±{std:.3f}"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    sessions = discover_sessions()
    if not sessions:
        print(f"No processed sessions found in {DERIV_ANALYSIS}")
        print("Run scripts/02_lcmv.py first.")
        return

    print(f"\nmcLCMV results table  ({len(sessions)} session(s))\n")

    hdr = (f"{'Subject':<10} {'Session':<14} "
           f"{'Separ.(°)':>12} {'UT Comp.':>13} {'BL Comp.':>13} "
           f"{'Null Gain':>13} {'|α|':>6} {'Δr':>7} {'E Rat.':>7}")
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)

    for sub, ses in sessions:
        try:
            r = load_session(sub, ses)
        except Exception as e:
            print(f"  {sub:<10} {ses:<14}  [ERROR: {e}]")
            continue

        m = compute_all_metrics(r)
        phase = r.get("meta", {}).get("cycle_phase", "")

        print(
            f"{sub:<10} {ses:<14} "
            f"  {fmt_pm(m['sep_mean'], m['sep_std']):>12} "
            f"  {fmt_pm3(m['ut_comp_mean'], m['ut_comp_std']):>13} "
            f"  {fmt_pm3(m['bl_comp_mean'], m['bl_comp_std']):>13} "
            f"  {fmt_pm3(m['ng_mean'], m['ng_std']):>13} "
            f"  {m['alpha']:>5.2f} "
            f"  {m['delta_r']:>6.3f} "
            f"  {m['energy_ratio']:>6.3f}"
            + (f"  ({phase})" if phase else "")
        )

    print(sep)


if __name__ == "__main__":
    main()
