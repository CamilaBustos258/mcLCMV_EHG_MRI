"""MRI preprocessing — surface extraction, electrode clustering, normalisation."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.cluster import DBSCAN

from mclcmv.config import DBSCAN_EPS_CM, DBSCAN_MIN_SAMPLES, EXPECTED_N_ELECTRODES

logger = logging.getLogger(__name__)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalise a 3-D volume to [0, 1]."""
    vmin, vmax = volume.min(), volume.max()
    if vmax == vmin:
        return np.zeros_like(volume, dtype=float)
    return (volume - vmin) / (vmax - vmin)


def extract_surface(
    mask: np.ndarray,
    voxel_sizes: np.ndarray,
    level: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run marching cubes on a binary mask and return vertices in cm.

    Parameters
    ----------
    mask        : 3-D binary (or float) array — values > level define the surface
    voxel_sizes : [3] voxel dimensions in mm
    level       : iso-surface threshold (default 0.5 for binary masks)

    Returns
    -------
    verts_cm : np.ndarray, shape (N, 3) — vertex coordinates in centimetres
    faces    : np.ndarray, shape (M, 3) — triangle face indices
    """
    verts, faces, _, _ = measure.marching_cubes(mask, level=level)
    verts_cm = (verts * voxel_sizes) / 10.0  # mm → cm
    return verts_cm, faces


def extract_electrode_coords(
    electrodes_mask: np.ndarray,
    voxel_sizes: np.ndarray,
    expected_n: int = EXPECTED_N_ELECTRODES,
    eps_cm: float = DBSCAN_EPS_CM,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """Cluster the electrode mask surface into individual electrode centroids.

    Uses DBSCAN on the surface vertices to separate the electrode blobs and
    returns one centroid per electrode (in cm).

    Parameters
    ----------
    electrodes_mask : 3-D binary mask of all electrode markers
    voxel_sizes     : [3] voxel dimensions in mm
    expected_n      : expected electrode count — a warning is issued if different
    eps_cm          : DBSCAN neighbourhood radius in cm
    min_samples     : DBSCAN minimum cluster size

    Returns
    -------
    electrode_coords : np.ndarray, shape (n_electrodes, 3) in cm
    """
    verts_cm, _ = extract_surface(electrodes_mask, voxel_sizes, level=0.5)
    clustering = DBSCAN(eps=eps_cm, min_samples=min_samples).fit(verts_cm)
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels != -1])

    coords = np.array(
        [verts_cm[labels == lbl].mean(axis=0) for lbl in unique_labels]
    )

    if len(coords) != expected_n:
        warnings.warn(
            f"Expected {expected_n} electrode clusters but found {len(coords)}. "
            "Check the electrodes mask quality.",
            stacklevel=2,
        )
    logger.info("Extracted %d electrode coordinates.", len(coords))
    return coords


def subsample_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    max_verts: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample a surface mesh to at most ``max_verts`` vertices.

    If the surface already has fewer vertices than ``max_verts``, it is returned
    unchanged. This is used to keep the steering dictionary build time tractable
    without substantially affecting DOA accuracy.

    Returns
    -------
    verts_sub : subsampled vertex array
    faces_sub : faces referencing only the subsampled vertices (approximate — uses
                kd-tree remapping, not exact topology preservation)
    """
    if len(verts) <= max_verts:
        return verts, faces

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(verts), size=max_verts, replace=False)
    verts_sub = verts[idx]

    # Remap faces: replace each original vertex index with the nearest subsampled one
    tree = cKDTree(verts_sub)
    _, new_idx = tree.query(verts[faces.reshape(-1)])
    faces_sub = new_idx.reshape(-1, 3)
    # Remove degenerate faces (all three corners collapsed to same vertex)
    faces_sub = faces_sub[
        (faces_sub[:, 0] != faces_sub[:, 1])
        & (faces_sub[:, 1] != faces_sub[:, 2])
        & (faces_sub[:, 0] != faces_sub[:, 2])
    ]
    return verts_sub, faces_sub
