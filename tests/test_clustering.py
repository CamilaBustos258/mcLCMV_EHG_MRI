"""Unit tests for beamforming/clustering.py."""

import numpy as np
import pytest

from mclcmv.beamforming.clustering import (
    build_cluster_representatives,
    cluster_compactness,
    cluster_to_steering_vector,
    spherical_kmeans,
    whiten_subspace,
)

RNG = np.random.default_rng(42)


class TestSphericalKmeans:
    def _unit_vectors(self, n, seed=0):
        v = np.random.default_rng(seed).standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_label_shape(self):
        vecs = self._unit_vectors(60)
        labels, _ = spherical_kmeans(vecs, k=3)
        assert labels.shape == (60,)

    def test_centroid_shape(self):
        vecs = self._unit_vectors(60)
        _, centroids = spherical_kmeans(vecs, k=3)
        assert centroids.shape == (3, 3)

    def test_labels_in_valid_range(self):
        vecs = self._unit_vectors(50)
        k = 4
        labels, _ = spherical_kmeans(vecs, k=k)
        assert set(labels).issubset(set(range(k)))

    def test_centroids_are_unit_norm(self):
        vecs = self._unit_vectors(40)
        _, centroids = spherical_kmeans(vecs, k=3)
        norms = np.linalg.norm(centroids, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_separable_clusters_correctly_assigned(self):
        # Build three tight, well-separated clusters near the coordinate axes
        rng = np.random.default_rng(7)
        poles = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        parts = []
        for pole in poles:
            pts = rng.standard_normal((20, 3)) * 0.05 + pole
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            parts.append(pts)
        vecs = np.vstack(parts)  # 60 vectors, 3 ground-truth clusters

        labels, _ = spherical_kmeans(vecs, k=3, seed=0)

        # Each ground-truth group must map to exactly one label
        for i in range(3):
            group = labels[i * 20: (i + 1) * 20]
            assert len(np.unique(group)) == 1, (
                f"Ground-truth cluster {i} was split across labels"
            )

    def test_k_equals_n_all_singletons(self):
        vecs = self._unit_vectors(5)
        labels, _ = spherical_kmeans(vecs, k=5, seed=0)
        assert len(np.unique(labels)) == 5

    def test_reproducible_with_seed(self):
        vecs = self._unit_vectors(30)
        labels1, _ = spherical_kmeans(vecs, k=3, seed=99)
        labels2, _ = spherical_kmeans(vecs, k=3, seed=99)
        assert np.array_equal(labels1, labels2)


class TestClusterToSteeringVector:
    def test_output_shape(self):
        M, G = 8, 30
        D = RNG.standard_normal((M, G)) + 1j * RNG.standard_normal((M, G))
        # normalise columns
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        mask = np.zeros(G, dtype=bool)
        mask[:10] = True
        sv = cluster_to_steering_vector(mask, D)
        assert sv.shape == (M,)

    def test_output_unit_norm(self):
        M, G = 6, 20
        rng = np.random.default_rng(1)
        D = rng.standard_normal((M, G)) + 1j * rng.standard_normal((M, G))
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        mask = rng.random(G) > 0.4
        if not mask.any():
            mask[0] = True
        sv = cluster_to_steering_vector(mask, D)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_single_member_returns_that_column(self):
        # with one member, the representative must equal that steering vector
        M = 5
        rng = np.random.default_rng(2)
        D = rng.standard_normal((M, 4)) + 0j
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        mask = np.array([False, True, False, False])
        sv = cluster_to_steering_vector(mask, D)
        assert np.allclose(np.abs(sv), np.abs(D[:, 1]), atol=1e-12)


class TestBuildClusterRepresentatives:
    def test_output_shapes(self):
        M, G, k = 8, 50, 3
        rng = np.random.default_rng(0)
        doa = rng.standard_normal((G, 3))
        doa /= np.linalg.norm(doa, axis=1, keepdims=True)
        D = rng.standard_normal((M, G)) + 1j * rng.standard_normal((M, G))
        D /= np.linalg.norm(D, axis=0, keepdims=True)

        S, labels, centroids = build_cluster_representatives(doa, D, k=k)
        assert S.shape == (M, k)
        assert labels.shape == (G,)
        assert centroids.shape == (k, 3)

    def test_representative_columns_are_unit_norm(self):
        M, G, k = 6, 40, 3
        rng = np.random.default_rng(3)
        doa = rng.standard_normal((G, 3))
        doa /= np.linalg.norm(doa, axis=1, keepdims=True)
        D = rng.standard_normal((M, G)) + 1j * rng.standard_normal((M, G))
        D /= np.linalg.norm(D, axis=0, keepdims=True)

        S, _, _ = build_cluster_representatives(doa, D, k=k)
        col_norms = np.linalg.norm(S, axis=0)
        assert np.allclose(col_norms, 1.0, atol=1e-10)


class TestClusterCompactness:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        doa = rng.standard_normal((30, 3))
        doa /= np.linalg.norm(doa, axis=1, keepdims=True)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        c = cluster_compactness(doa, labels, k=3)
        assert c.shape == (3,)

    def test_tight_cluster_lower_compactness(self):
        # cluster 0: tight (low angular spread), cluster 1: spread out
        rng = np.random.default_rng(0)
        pole = np.array([0.0, 0.0, 1.0])
        tight = rng.standard_normal((20, 3)) * 0.01 + pole
        tight /= np.linalg.norm(tight, axis=1, keepdims=True)
        spread = rng.standard_normal((20, 3))
        spread /= np.linalg.norm(spread, axis=1, keepdims=True)
        doa = np.vstack([tight, spread])
        labels = np.array([0] * 20 + [1] * 20)
        c = cluster_compactness(doa, labels, k=2)
        assert c[0] < c[1]

    def test_singleton_cluster_zero_compactness(self):
        rng = np.random.default_rng(0)
        doa = rng.standard_normal((5, 3))
        doa /= np.linalg.norm(doa, axis=1, keepdims=True)
        labels = np.array([0, 1, 2, 3, 4])
        c = cluster_compactness(doa, labels, k=5)
        assert np.allclose(c, 0.0)


class TestWhitenSubspace:
    def test_svd_columns_orthonormal(self):
        M, K = 8, 3
        rng = np.random.default_rng(0)
        S = rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))
        Q = whiten_subspace(S, method="svd")
        # Q^H Q should be identity
        gram = Q.conj().T @ Q
        assert np.allclose(gram, np.eye(Q.shape[1]), atol=1e-10)

    def test_qr_columns_orthonormal(self):
        M, K = 8, 3
        rng = np.random.default_rng(1)
        S = rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))
        Q = whiten_subspace(S, method="qr")
        gram = Q.conj().T @ Q
        assert np.allclose(gram, np.eye(Q.shape[1]), atol=1e-10)

    def test_svd_spans_same_column_space(self):
        M, K = 8, 2
        rng = np.random.default_rng(2)
        S = rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))
        Q = whiten_subspace(S, method="svd")
        # Every column of S should lie in the column space of Q:
        # residual of projection of S onto Q should be ~0
        P = Q @ Q.conj().T          # projector onto col(Q)
        residual = np.linalg.norm(S - P @ S)
        assert residual == pytest.approx(0.0, abs=1e-10)

    def test_invalid_method_raises(self):
        S = np.eye(4, 2)
        with pytest.raises(ValueError):
            whiten_subspace(S, method="unknown")
