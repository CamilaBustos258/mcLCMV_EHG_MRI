"""Windowed multicluster LCMV beamformer with Bayesian covariance shrinkage.

Implements Sections 2.2 C–D of the paper:

    C. Windowed LCMV with null constraints
       - Dual beamformers: uterus (pass) + bladder (null) and vice versa
       - Each beamformer uses its own band's covariance matrix
       - Bayesian diagonal shrinkage: R̃_w = (1−β)R_w + β·diag(R_w)
       - Constraint matrix: C = [S^(U)_pass | Q^(B)_null] ∈ C^{M×L}
       - LCMV solution: w_w = R̃_w⁻¹ C (C^H R̃_w⁻¹ C)⁻¹ f

    D. Overlap-add reconstruction
       - Hann-tapered windows, 75% overlap (COLA satisfied)
       - Scalar leakage subtraction on bladder estimate

Reference: Bustos-Vivas et al., "Disentangling uterine and bladder activity
in paired EHG-MRI data using anatomy-guided multicluster beamforming", 2025.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


# ── Covariance ─────────────────────────────────────────────────────────────

def sample_covariance(segment: np.ndarray) -> np.ndarray:
    """Compute the sample covariance matrix for a data segment.

    Matches the paper formula R_w = (1/N_w) Σ x[n] x[n]^H — no mean
    subtraction, consistent with the zero-mean assumption after bandpass
    filtering.

    Parameters
    ----------
    segment : (M, N_w) complex or real data window

    Returns
    -------
    R : (M, M) Hermitian sample covariance
    """
    return (segment @ segment.conj().T) / segment.shape[1]


def shrink_covariance(R: np.ndarray, beta: float = 0.10) -> np.ndarray:
    """Apply Bayesian diagonal shrinkage regularisation.

    R̃ = (1 − β) R + β · diag(R)

    Parameters
    ----------
    R    : (M, M) sample covariance
    beta : shrinkage factor ∈ [0, 1]

    Returns
    -------
    R_tilde : (M, M) regularised covariance
    """
    return (1.0 - beta) * R + beta * np.diag(np.diag(R).real)


def add_diagonal_loading(R: np.ndarray, load_factor: float = 1e-3) -> np.ndarray:
    """Add scaled identity loading for additional numerical stability.

    R ← R + load_factor × trace(R)/M × I

    Parameters
    ----------
    R           : (M, M) covariance (possibly already shrinkage-regularised)
    load_factor : scaling relative to mean eigenvalue
    """
    M = R.shape[0]
    return R + load_factor * (np.trace(R).real / M) * np.eye(M, dtype=R.dtype)


# ── Constraint matrix & LCMV solution ─────────────────────────────────────

def build_constraint_matrix(
    S_pass: np.ndarray,
    S_null: np.ndarray,
    n_null: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the constraint matrix C and target vector f.

    C = [ s^(U)_pass | s^(B)_null_1 | … | s^(B)_null_{n_null} ] ∈ C^{M × (1 + n_null)}
    f = [1, 0, …, 0]^T ∈ C^{1 + n_null}

    The pass constraint is a single unity-gain vector obtained by averaging
    and re-normalising the K_U cluster representatives (pass_mode="mean").
    Keeping L = 1 + n_null = 3 constraints for
    M = 8 channels (5 degrees of freedom for variance minimisation).

    The n_null null columns are the first n_null raw cluster steering vectors
    of the opposing organ (S_null[:,0], S_null[:,1], …). 

    Parameters
    ----------
    S_pass : (M, K_U) uterus cluster representative steering vectors
    S_null : (M, K_B) bladder cluster representative steering vectors
    n_null : number of null directions to retain (default 2)

    Returns
    -------
    C : (M, 1 + n_null) constraint matrix
    f : (1 + n_null,) target vector
    """
    # Single pass vector: normalised mean of the K_U cluster representatives
    s_pass = S_pass.mean(axis=1)
    s_pass = s_pass / (np.linalg.norm(s_pass) + 1e-12)
    C_pass = s_pass[:, None]                     # (M, 1)

    # Null columns: first n_null raw cluster steering vectors of the
    # opposing organ.  Each column is already unit-normalised (from
    # cluster_to_steering_vector).  Using raw vectors leaves the remaining
    # K − n_null cluster directions unconstrained, which is intentional.
    K = S_null.shape[1]
    n_null_eff = min(n_null, K)
    Q_null = S_null[:, :n_null_eff]              # (M, n_null_eff)

    C = np.hstack([C_pass, Q_null])              # (M, 1 + n_null_eff)
    f = np.zeros(C.shape[1], dtype=complex)
    f[0] = 1.0                                   # unity gain on the pass direction
    return C, f


def lcmv_weights(
    R_reg: np.ndarray,
    C: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """Solve the LCMV beamforming problem for one window.

    w_w = R̃⁻¹ C (C^H R̃⁻¹ C)⁻¹ f,   s.t. C^H w_w = f

    Parameters
    ----------
    R_reg : (M, M) regularised covariance
    C     : (M, L) constraint matrix
    f     : (L,) target vector

    Returns
    -------
    w : (M,) complex beamformer weight vector
    """
    R_inv_C = np.linalg.solve(R_reg, C)   # (M, L) — avoids forming full R^{-1}
    gram = C.conj().T @ R_inv_C           # (L, L)
    return R_inv_C @ np.linalg.solve(gram, f)


def constraint_residual(C: np.ndarray, w: np.ndarray, f: np.ndarray) -> float:
    """Return ‖C^H w − f‖_∞ — should be ≪ 1 for valid constraints."""
    return float(np.max(np.abs(C.conj().T @ w - f)))


# ── Overlap-add reconstruction ─────────────────────────────────────────────

def hann_window(n: int) -> np.ndarray:
    """Return a Hann window of length n."""
    return np.hanning(n).astype(float)


def run_windowed_lcmv(
    data_uterus_band: np.ndarray,
    data_bladder_band: np.ndarray,
    S_uterus: np.ndarray,
    S_bladder: np.ndarray,
    fs: float,
    window_size_sec: float = 120.0,
    step_size_sec: float = 30.0,
    beta: float = 0.10,
    diag_load: float = 1e-3,
    n_null: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Windowed dual-beamformer LCMV with overlap-add reconstruction.

    Runs two separate LCMV beamformers — one per organ — each using the other
    organ's cluster subspace as null constraints and its own band's covariance:

        w_ut: pass = S_uterus,  null = whitened(S_bladder),  R from uterus band
        w_bl: pass = S_bladder, null = whitened(S_uterus),   R from bladder band

    Parameters
    ----------
    data_uterus_band  : (M, N) uterus-band filtered EHG
    data_bladder_band : (M, N) bladder-band filtered EHG
    S_uterus          : (M, K_U) uterus cluster steering vectors
    S_bladder         : (M, K_B) bladder cluster steering vectors
    fs                : sampling frequency in Hz
    window_size_sec   : Hann window length in seconds (default 120 s)
    step_size_sec     : step between windows in seconds (default 30 s → 75% overlap)
    beta              : Bayesian shrinkage factor
    diag_load         : diagonal loading factor
    n_null            : null directions retained per organ (default 2)

    Returns
    -------
    y_uterus    : (N,) reconstructed uterus time series (overlap-add)
    y_bladder   : (N,) reconstructed bladder time series (overlap-add, pre-leakage)
    diagnostics : list of per-window dicts with keys:
                  window_idx, t0, t1,
                  ut_pass_residual, ut_null_residual,
                  bl_pass_residual, bl_null_residual,
                  alpha_leakage, w_ut, w_bl
    """
    N = data_uterus_band.shape[1]
    win_len = int(window_size_sec * fs)
    step = int(step_size_sec * fs)
    n_windows = max(0, (N - win_len) // step + 1)

    hann = hann_window(win_len)
    y_uterus = np.zeros(N)
    y_bladder = np.zeros(N)
    norm_accum = np.zeros(N)

    # Build both constraint matrices once — they are data-independent.
    # Each C has shape (M, 1 + n_null): one mean pass vector + n_null null dirs.
    C_ut, f_ut = build_constraint_matrix(S_uterus,  S_bladder, n_null=n_null)
    C_bl, f_bl = build_constraint_matrix(S_bladder, S_uterus,  n_null=n_null)

    diagnostics = []

    for wi in tqdm(range(n_windows), desc="LCMV windows", leave=False):
        start = wi * step
        end = start + win_len

        seg_u = data_uterus_band[:, start:end]
        seg_b = data_bladder_band[:, start:end]

        # Uterus beamformer — covariance from uterus-band segment
        R_ut = sample_covariance(seg_u)
        R_ut = shrink_covariance(R_ut, beta=beta)
        R_ut = add_diagonal_loading(R_ut, load_factor=diag_load)
        w_ut = lcmv_weights(R_ut, C_ut, f_ut)

        # Bladder beamformer — covariance from bladder-band segment
        R_bl = sample_covariance(seg_b)
        R_bl = shrink_covariance(R_bl, beta=beta)
        R_bl = add_diagonal_loading(R_bl, load_factor=diag_load)
        w_bl = lcmv_weights(R_bl, C_bl, f_bl)

        # Constraint residuals: column 0 = pass, columns 1: = null
        ut_pass_res = constraint_residual(C_ut[:, :1], w_ut, f_ut[:1])
        ut_null_res = float(np.max(np.abs(C_ut[:, 1:].conj().T @ w_ut)))
        bl_pass_res = constraint_residual(C_bl[:, :1], w_bl, f_bl[:1])
        bl_null_res = float(np.max(np.abs(C_bl[:, 1:].conj().T @ w_bl)))

        # Per-window beamformer outputs (tapered, pre-WOLA)
        y_ut_w = (w_ut.conj() @ seg_u).real
        y_bl_w = (w_bl.conj() @ seg_b).real

        # Per-window leakage coefficient α_w = <y_bl_w, y_ut_w> / <y_ut_w, y_ut_w>
        denom_w = float(np.dot(y_ut_w, y_ut_w))
        alpha_w = float(np.dot(y_bl_w, y_ut_w)) / denom_w if denom_w > 1e-12 else 0.0

        diagnostics.append({
            "window_idx": wi,
            "t0": int(start),
            "t1": int(end),
            "ut_pass_residual": ut_pass_res,
            "ut_null_residual": ut_null_res,
            "bl_pass_residual": bl_pass_res,
            "bl_null_residual": bl_null_res,
            "alpha_leakage": alpha_w,
            "w_ut": w_ut.copy(),
            "w_bl": w_bl.copy(),
        })

        # Overlap-add accumulation
        y_uterus[start:end]  += y_ut_w * hann
        y_bladder[start:end] += y_bl_w * hann
        norm_accum[start:end] += hann ** 2

    # WOLA normalisation
    safe_norm = np.where(norm_accum < 1e-12, 1.0, norm_accum)
    y_uterus  /= safe_norm
    y_bladder /= safe_norm

    return y_uterus, y_bladder, diagnostics


def leakage_subtraction(
    y_uterus: np.ndarray,
    y_bladder: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Remove coherent uterus leakage from the bladder estimate (global OLS).

    Computes a scalar leakage coefficient as a single ordinary-least-squares
    projection over the full reconstructed signals:

        α = <ŷ_B, ŷ_U> / <ŷ_U, ŷ_U>

    and returns the corrected bladder signal  ŷ_B − α ŷ_U.

    Note
    ----
    The pipeline (``scripts/02_lcmv.py``) uses the more robust
    **per-window median** of ``diagnostics["alpha_leakage"]`` values instead,
    and falls back to this function only when no per-window values are
    available.  The paper definition matches the per-window median approach.

    Parameters
    ----------
    y_uterus  : (N,) reconstructed uterus signal
    y_bladder : (N,) reconstructed bladder signal (before leakage removal)

    Returns
    -------
    y_bladder_clean : (N,) bladder signal after leakage subtraction
    alpha           : global OLS leakage coefficient
    """
    denom = float(np.dot(y_uterus, y_uterus))
    alpha = float(np.dot(y_bladder, y_uterus)) / denom if denom > 1e-12 else 0.0
    return y_bladder - alpha * y_uterus, alpha
