"""Unit tests for beamforming/lcmv.py."""

import numpy as np
import pytest

from mclcmv.beamforming.lcmv import (
    add_diagonal_loading,
    build_constraint_matrix,
    constraint_residual,
    lcmv_weights,
    leakage_subtraction,
    sample_covariance,
    shrink_covariance,
)

RNG = np.random.default_rng(0)


# ── Helpers ────────────────────────────────────────────────────────────────

def _psd_matrix(M: int, seed: int = 0) -> np.ndarray:
    """Return a random positive-definite real M×M matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M))
    return A @ A.T + np.eye(M) * 0.1


def _psd_complex(M: int, seed: int = 0) -> np.ndarray:
    """Return a random positive-definite Hermitian complex M×M matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    R = A @ A.conj().T
    R += np.eye(M) * 0.1
    return R


# ── sample_covariance ──────────────────────────────────────────────────────

class TestSampleCovariance:
    def test_shape(self):
        M, N = 8, 500
        seg = RNG.standard_normal((M, N))
        R = sample_covariance(seg)
        assert R.shape == (M, M)

    def test_hermitian(self):
        M, N = 6, 200
        seg = RNG.standard_normal((M, N)) + 1j * RNG.standard_normal((M, N))
        R = sample_covariance(seg)
        assert np.allclose(R, R.conj().T, atol=1e-12)

    def test_positive_semidefinite(self):
        M, N = 5, 300
        seg = RNG.standard_normal((M, N))
        R = sample_covariance(seg)
        eigvals = np.linalg.eigvalsh(R)
        assert np.all(eigvals >= -1e-12)

    def test_matches_manual_formula(self):
        M, N = 4, 100
        seg = RNG.standard_normal((M, N))
        R = sample_covariance(seg)
        R_manual = (seg @ seg.T) / N
        assert np.allclose(R, R_manual, atol=1e-14)

    def test_no_mean_subtraction(self):
        # Paper formula has no mean centering; verify by using a segment with
        # a non-zero DC offset — sample_covariance should not remove it.
        M, N = 3, 200
        seg = RNG.standard_normal((M, N)) + 5.0   # large DC
        R_ours = sample_covariance(seg)
        R_centered = np.cov(seg)                  # scipy/numpy cov centres
        # They should differ when DC is large
        assert not np.allclose(R_ours, R_centered, atol=1.0)


# ── shrink_covariance ──────────────────────────────────────────────────────

class TestShrinkCovariance:
    def test_beta_zero_returns_input(self):
        R = _psd_matrix(5)
        assert np.allclose(shrink_covariance(R, beta=0.0), R)

    def test_beta_one_returns_diagonal(self):
        R = _psd_matrix(5)
        result = shrink_covariance(R, beta=1.0)
        assert np.allclose(result, np.diag(np.diag(R)))

    def test_diagonal_always_preserved(self):
        R = _psd_matrix(6)
        for beta in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = shrink_covariance(R, beta=beta)
            assert np.allclose(np.diag(result).real, np.diag(R).real, atol=1e-14)

    def test_off_diagonal_scaled_by_one_minus_beta(self):
        R = _psd_matrix(4, seed=1)
        beta = 0.3
        result = shrink_covariance(R, beta=beta)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert result[i, j] == pytest.approx((1 - beta) * R[i, j], rel=1e-12)

    def test_output_still_hermitian_for_hermitian_input(self):
        R = _psd_complex(5)
        result = shrink_covariance(R, beta=0.2)
        assert np.allclose(result, result.conj().T, atol=1e-12)

    def test_positive_semidefinite_preserved(self):
        R = _psd_matrix(6)
        result = shrink_covariance(R, beta=0.15)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals >= -1e-12)


# ── add_diagonal_loading ───────────────────────────────────────────────────

class TestAddDiagonalLoading:
    def test_diagonal_increases(self):
        R = _psd_matrix(4)
        R_loaded = add_diagonal_loading(R, load_factor=1e-3)
        assert np.all(np.diag(R_loaded) > np.diag(R))

    def test_off_diagonal_unchanged(self):
        R = _psd_matrix(4)
        R_loaded = add_diagonal_loading(R, load_factor=1e-3)
        mask = ~np.eye(4, dtype=bool)
        assert np.allclose(R_loaded[mask], R[mask])


# ── build_constraint_matrix ────────────────────────────────────────────────

class TestBuildConstraintMatrix:
    def test_shape(self):
        # C always has 1 pass column (mean of K_U clusters) + n_null null cols
        M, K_U, K_B, n_null = 8, 3, 3, 2
        rng = np.random.default_rng(0)
        S_pass = rng.standard_normal((M, K_U)) + 0j
        S_null = rng.standard_normal((M, K_B)) + 0j
        C, f = build_constraint_matrix(S_pass, S_null, n_null=n_null)
        assert C.shape == (M, 1 + n_null)
        assert f.shape == (1 + n_null,)

    def test_target_vector_structure(self):
        M, K_U, n_null = 8, 3, 2
        rng = np.random.default_rng(1)
        S_pass = rng.standard_normal((M, K_U)) + 0j
        S_null = rng.standard_normal((M, 3)) + 0j
        _, f = build_constraint_matrix(S_pass, S_null, n_null=n_null)
        # First entry = 1 (pass), rest = 0 (null)
        assert f[0] == pytest.approx(1.0)
        assert np.allclose(f[1:], 0.0)

    def test_pass_column_is_unit_norm(self):
        # The single pass vector must be ℓ₂-normalised
        M, K_U, n_null = 8, 3, 2
        rng = np.random.default_rng(3)
        S_pass = rng.standard_normal((M, K_U)) + 0j
        S_null = rng.standard_normal((M, 3)) + 0j
        C, _ = build_constraint_matrix(S_pass, S_null, n_null=n_null)
        assert np.linalg.norm(C[:, 0]) == pytest.approx(1.0, abs=1e-12)

    def test_null_columns_are_raw_cluster_vectors(self):
        # The null columns should be the first n_null columns of S_null
        # (raw cluster steering vectors, not SVD-compressed).
        M, K_U, K_B, n_null = 8, 2, 3, 2
        rng = np.random.default_rng(2)
        S_pass = rng.standard_normal((M, K_U)) + 0j
        S_null = rng.standard_normal((M, K_B)) + 0j
        C, _ = build_constraint_matrix(S_pass, S_null, n_null=n_null)
        Q = C[:, 1:]   # columns 1: are the null directions (raw cluster vectors)
        # Null columns should equal the first n_null columns of S_null exactly
        assert np.allclose(Q, S_null[:, :n_null], atol=1e-12)


# ── lcmv_weights ───────────────────────────────────────────────────────────

class TestLcmvWeights:
    def test_output_shape(self):
        M, L = 8, 3
        R = _psd_matrix(M)
        rng = np.random.default_rng(0)
        C = rng.standard_normal((M, L)) + 1j * rng.standard_normal((M, L))
        f = np.zeros(L, dtype=complex)
        f[0] = 1.0
        w = lcmv_weights(R.astype(complex), C, f)
        assert w.shape == (M,)

    def test_constraint_satisfied(self):
        # The fundamental LCMV property: C^H w = f exactly
        M, L = 8, 3
        rng = np.random.default_rng(1)
        R = _psd_complex(M, seed=1)
        C = rng.standard_normal((M, L)) + 1j * rng.standard_normal((M, L))
        f = np.array([1.0, 0.0, 0.0], dtype=complex)
        w = lcmv_weights(R, C, f)
        residual = C.conj().T @ w - f
        assert np.allclose(residual, 0.0, atol=1e-8)

    def test_constraint_satisfied_multi_pass(self):
        # Multiple unity-gain pass constraints
        M, K_U, n_null = 8, 3, 2
        rng = np.random.default_rng(2)
        R = _psd_complex(M, seed=2)
        S_pass = rng.standard_normal((M, K_U)) + 1j * rng.standard_normal((M, K_U))
        S_null = rng.standard_normal((M, 4)) + 1j * rng.standard_normal((M, 4))
        C, f = build_constraint_matrix(S_pass, S_null, n_null=n_null)
        w = lcmv_weights(R, C, f)
        assert np.allclose(C.conj().T @ w, f, atol=1e-8)

    def test_single_pass_direction_unity_gain(self):
        # With a single pass direction, w^H s_pass = 1
        M = 8
        rng = np.random.default_rng(3)
        R = _psd_complex(M, seed=3)
        s_pass = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        s_pass /= np.linalg.norm(s_pass)
        C = s_pass[:, None]
        f = np.array([1.0 + 0j])
        w = lcmv_weights(R, C, f)
        gain = float(np.abs(w.conj() @ s_pass))
        assert gain == pytest.approx(1.0, abs=1e-8)

    def test_null_direction_suppressed(self):
        # Null constraint: beamformer gain along null direction ≈ 0
        M = 8
        rng = np.random.default_rng(4)
        R = _psd_complex(M, seed=4)
        s_pass = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        s_pass /= np.linalg.norm(s_pass)
        # Construct a null direction orthogonal to s_pass
        s_null = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        s_null -= s_pass * (s_pass.conj() @ s_null)
        s_null /= np.linalg.norm(s_null)
        C = np.column_stack([s_pass, s_null])
        f = np.array([1.0, 0.0], dtype=complex)
        w = lcmv_weights(R, C, f)
        null_gain = float(np.abs(w.conj() @ s_null))
        assert null_gain == pytest.approx(0.0, abs=1e-8)


# ── constraint_residual ────────────────────────────────────────────────────

class TestConstraintResidual:
    def test_exact_constraint_zero_residual(self):
        M, L = 6, 2
        rng = np.random.default_rng(0)
        R = _psd_complex(M)
        C = rng.standard_normal((M, L)) + 1j * rng.standard_normal((M, L))
        f = np.array([1.0, 0.0], dtype=complex)
        w = lcmv_weights(R, C, f)
        res = constraint_residual(C, w, f)
        assert res == pytest.approx(0.0, abs=1e-8)

    def test_violated_constraint_nonzero_residual(self):
        M, L = 6, 2
        rng = np.random.default_rng(1)
        C = rng.standard_normal((M, L)) + 0j
        w = rng.standard_normal(M) + 0j
        f = np.array([1.0, 0.0], dtype=complex)
        res = constraint_residual(C, w, f)
        assert res > 0.0


# ── leakage_subtraction ───────────────────────────────────────────────────

class TestLeakageSubtraction:
    def test_alpha_formula(self):
        y_u = np.array([1.0, 2.0, 3.0, 4.0])
        y_b = 2.5 * y_u
        _, alpha = leakage_subtraction(y_u, y_b)
        assert alpha == pytest.approx(2.5, rel=1e-12)

    def test_fully_correlated_gives_zero_residual(self):
        y_u = np.array([1.0, -1.0, 2.0, -2.0])
        y_b = 3.0 * y_u
        y_clean, alpha = leakage_subtraction(y_u, y_b)
        assert alpha == pytest.approx(3.0, rel=1e-12)
        assert np.allclose(y_clean, 0.0, atol=1e-12)

    def test_orthogonal_signals_unchanged(self):
        # <y_u, y_b> = 0 → alpha = 0 → y_clean = y_b
        y_u = np.array([1.0, -1.0, 1.0, -1.0])
        y_b = np.array([1.0,  1.0, 1.0,  1.0])
        assert np.dot(y_u, y_b) == pytest.approx(0.0)
        y_clean, alpha = leakage_subtraction(y_u, y_b)
        assert alpha == pytest.approx(0.0, abs=1e-14)
        assert np.allclose(y_clean, y_b)

    def test_zero_uterus_no_crash_alpha_zero(self):
        y_u = np.zeros(10)
        y_b = np.ones(10)
        y_clean, alpha = leakage_subtraction(y_u, y_b)
        assert alpha == pytest.approx(0.0)
        assert np.allclose(y_clean, y_b)

    def test_energy_decreases_after_subtraction(self):
        rng = np.random.default_rng(5)
        y_u = rng.standard_normal(200)
        y_b = 0.6 * y_u + rng.standard_normal(200) * 0.1
        y_clean, _ = leakage_subtraction(y_u, y_b)
        assert np.linalg.norm(y_clean) < np.linalg.norm(y_b)

    def test_correlation_reduced_after_subtraction(self):
        rng = np.random.default_rng(6)
        y_u = rng.standard_normal(300)
        y_b = 0.8 * y_u + rng.standard_normal(300) * 0.2
        y_clean, _ = leakage_subtraction(y_u, y_b)
        corr_before = np.dot(y_b, y_u) / (np.linalg.norm(y_b) * np.linalg.norm(y_u))
        corr_after  = np.dot(y_clean, y_u) / (np.linalg.norm(y_clean) * np.linalg.norm(y_u) + 1e-12)
        assert abs(corr_after) < abs(corr_before)

    def test_returns_alpha_float(self):
        y_u = np.ones(10)
        y_b = np.ones(10) * 2.0
        _, alpha = leakage_subtraction(y_u, y_b)
        assert isinstance(alpha, float)
