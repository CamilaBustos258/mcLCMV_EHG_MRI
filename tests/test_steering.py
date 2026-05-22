"""Unit tests for forward/steering.py."""

import numpy as np
import pytest

from mclcmv.forward.steering import (
    build_steering_dictionary,
    compute_doa_vectors,
    compute_steering_vector,
)

RNG = np.random.default_rng(0)


class TestComputeSteeringVector:
    def test_output_is_unit_norm(self):
        elec = RNG.standard_normal((8, 3))
        vertex = RNG.standard_normal(3)
        sv = compute_steering_vector(elec, vertex)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-12)

    def test_output_is_complex(self):
        elec = RNG.standard_normal((4, 3))
        sv = compute_steering_vector(elec, np.zeros(3))
        assert np.iscomplexobj(sv)

    def test_output_shape(self):
        M = 8
        elec = RNG.standard_normal((M, 3))
        sv = compute_steering_vector(elec, np.zeros(3))
        assert sv.shape == (M,)

    def test_closer_electrode_larger_amplitude(self):
        # electrode at distance 1 should have larger weight than electrode at distance 3
        elec = np.array([[0.0, 0.0, 1.0],
                         [0.0, 0.0, 3.0]])
        vertex = np.zeros(3)
        sv = compute_steering_vector(elec, vertex, p=1.0)
        assert np.abs(sv[0]) > np.abs(sv[1])

    def test_exponent_p_increases_contrast(self):
        # larger p → more contrast between near and far electrodes
        elec = np.array([[0.0, 0.0, 1.0],
                         [0.0, 0.0, 4.0]])
        vertex = np.zeros(3)
        sv1 = compute_steering_vector(elec, vertex, p=1.0)
        sv2 = compute_steering_vector(elec, vertex, p=2.0)
        ratio_p1 = np.abs(sv1[0]) / np.abs(sv1[1])
        ratio_p2 = np.abs(sv2[0]) / np.abs(sv2[1])
        assert ratio_p2 > ratio_p1

    def test_electrode_at_vertex_no_crash(self):
        # electrode coincides with vertex — clipped distance must prevent div/0
        elec = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0]])
        sv = compute_steering_vector(elec, np.zeros(3))
        assert np.isfinite(sv).all()
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_all_equal_distances_gives_uniform_vector(self):
        # if every electrode is equidistant from the vertex the raw
        # weights are all equal → normalised sv has equal-magnitude components
        vertex = np.zeros(3)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        elec = np.column_stack([np.cos(angles), np.sin(angles), np.ones(8)])
        sv = compute_steering_vector(elec, vertex, p=1.0)
        assert np.allclose(np.abs(sv), np.abs(sv[0]), atol=1e-12)


class TestBuildSteeringDictionary:
    def test_shape(self):
        M, G = 8, 50
        elec = RNG.standard_normal((M, 3))
        verts = RNG.standard_normal((G, 3))
        D = build_steering_dictionary(elec, verts)
        assert D.shape == (M, G)

    def test_columns_are_unit_norm(self):
        M, G = 6, 30
        elec = RNG.standard_normal((M, 3))
        verts = RNG.standard_normal((G, 3))
        D = build_steering_dictionary(elec, verts)
        col_norms = np.linalg.norm(D, axis=0)
        assert np.allclose(col_norms, 1.0, atol=1e-12)

    def test_consistent_with_compute_steering_vector(self):
        M, G = 5, 10
        elec = RNG.standard_normal((M, 3))
        verts = RNG.standard_normal((G, 3))
        D = build_steering_dictionary(elec, verts, p=1.5)
        for gi in range(G):
            expected = compute_steering_vector(elec, verts[gi], p=1.5)
            assert np.allclose(D[:, gi], expected, atol=1e-12)


class TestComputeDOAVectors:
    def test_output_shape(self):
        verts = RNG.standard_normal((40, 3))
        center = np.zeros(3)
        doa = compute_doa_vectors(verts, center)
        assert doa.shape == (40, 3)

    def test_output_unit_norm(self):
        verts = RNG.standard_normal((20, 3)) + 5.0   # keep away from origin
        center = np.zeros(3)
        doa = compute_doa_vectors(verts, center)
        norms = np.linalg.norm(doa, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-12)

    def test_direction_correct(self):
        # vertex directly above the array center → DOA should point straight up
        center = np.array([0.0, 0.0, 0.0])
        verts = np.array([[0.0, 0.0, 5.0]])
        doa = compute_doa_vectors(verts, center)
        assert np.allclose(doa[0], [0.0, 0.0, 1.0], atol=1e-12)
