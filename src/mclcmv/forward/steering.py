"""MRI-derived forward model — steering vectors and steering dictionaries.

Implements the quasi-static bio-electric conduction model from the paper:

    a_i(r) ∝ ‖r_i − r‖^{−p},  p ∈ [1, 2]

After ℓ₂-normalisation, stacking per-vertex vectors yields organ-specific
steering dictionaries D_U ∈ C^{M×G_U} and D_B ∈ C^{M×G_B}.

Reference: Bustos-Vivas et al., "Disentangling uterine and bladder activity
in paired EHG-MRI data using anatomy-guided multicluster beamforming", 2025.
"""

from __future__ import annotations

import numpy as np


def compute_steering_vector(
    electrode_positions: np.ndarray,
    vertex: np.ndarray,
    p: float = 1.0,
) -> np.ndarray:
    """Compute an ℓ₂-normalised steering vector for one source location.

    Models quasi-static bio-electric conduction: signal amplitude decays as
    the inverse p-th power of distance from the source to each electrode.

    Parameters
    ----------
    electrode_positions : (M, 3) electrode coordinates in cm
    vertex              : (3,) candidate source location in cm
    p                   : distance-decay exponent; p=1 (default) or p=2

    Returns
    -------
    sv : (M,) complex, ℓ₂-normalised steering vector
    """
    distances = np.linalg.norm(electrode_positions - vertex, axis=1)
    distances = np.where(distances < 1e-10, 1e-10, distances)  # avoid division by zero
    raw = distances ** (-p) + 0j  # cast to complex for consistency with LCMV solver
    return raw / np.linalg.norm(raw)


def build_steering_dictionary(
    electrode_positions: np.ndarray,
    organ_vertices: np.ndarray,
    p: float = 1.0,
) -> np.ndarray:
    """Build the full steering dictionary for one organ.

    Parameters
    ----------
    electrode_positions : (M, 3) electrode coordinates in cm
    organ_vertices      : (G, 3) organ surface vertex coordinates in cm
    p                   : distance-decay exponent

    Returns
    -------
    D : (M, G) complex steering dictionary — each column is one steering vector
    """
    # (G, M, 3) → (G, M) distances, fully vectorised over all vertices
    diff = electrode_positions[np.newaxis, :, :] - organ_vertices[:, np.newaxis, :]
    distances = np.linalg.norm(diff, axis=2).clip(1e-10)   # (G, M)
    raw = distances ** (-p) + 0j                            # (G, M) complex
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / norms).T                                  # (M, G)


def compute_doa_vectors(
    organ_vertices: np.ndarray,
    array_center: np.ndarray,
) -> np.ndarray:
    """Map organ vertices to unit direction-of-arrival (DOA) vectors.

    Each vertex is expressed as a unit vector from the electrode array centre:

        d(r) = (r − c_array) / ‖r − c_array‖

    Parameters
    ----------
    organ_vertices : (G, 3) source candidate locations in cm
    array_center   : (3,) geometric centre of the electrode array in cm

    Returns
    -------
    doa : (G, 3) unit DOA vectors
    """
    diff = organ_vertices - array_center
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    return diff / norms
