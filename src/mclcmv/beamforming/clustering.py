"""Multi-cluster candidate selection via spherical k-means on DOA vectors.

Implements Section 2.2 (Multi-cluster Candidate Selection) of the paper:

    1. Map each organ vertex to a unit DOA vector.
    2. Cluster DOA vectors into K groups using spherical k-means.
    3. Compute cluster centroids and map each cluster back to sensor space
       to produce representative steering vectors s_k.

These cluster representatives form the pass (uterus) and null (bladder)
constraint matrices S^(U) and S^(B) for the LCMV solver.

Reference: Bustos-Vivas et al., 2025.
"""

from __future__ import annotations

import numpy as np


def spherical_kmeans(
    vectors: np.ndarray,
    k: int,
    n_iter: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means clustering on unit vectors.

    Parameters
    ----------
    vectors : (N, 3) unit direction vectors (DOAs)
    k       : number of clusters
    n_iter  : maximum iterations
    seed    : random seed for centroid initialisation

    Returns
    -------
    labels   : (N,) integer cluster assignments
    centroids: (k, 3) unit cluster centroids
    """
    rng = np.random.default_rng(seed)
    # Initialise centroids by sampling k vectors at random
    idx = rng.choice(len(vectors), size=k, replace=False)
    centroids = vectors[idx].copy()
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(n_iter):
        # Assignment: cosine similarity = dot product of unit vectors
        sims = vectors @ centroids.T      # (N, k)
        new_labels = np.argmax(sims, axis=1)

        # Update: recompute centroids as the normalised mean of members
        new_centroids = np.zeros_like(centroids)
        for ki in range(k):
            members = vectors[new_labels == ki]
            if len(members) == 0:
                new_centroids[ki] = centroids[ki]  # keep old centroid if empty
            else:
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean)
                new_centroids[ki] = mean / norm if norm > 1e-12 else centroids[ki]

        if np.array_equal(new_labels, labels):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids

    return labels, centroids


def cluster_to_steering_vector(
    cluster_mask: np.ndarray,
    steering_dict: np.ndarray,
) -> np.ndarray:
    """Compute a representative steering vector for one cluster.

    Averages the normalised steering vectors of all cluster members, then
    normalises the result (as in the paper):

        s_k = mean_{r_i ∈ C_k}( a(r_i) / ‖a(r_i)‖ )
              ──────────────────────────────────────────
              ‖ mean_{r_i ∈ C_k}( a(r_i) / ‖a(r_i)‖ ) ‖

    Parameters
    ----------
    cluster_mask  : (G,) boolean — True for vertices belonging to this cluster
    steering_dict : (M, G) complex steering dictionary for the organ

    Returns
    -------
    sv : (M,) complex representative steering vector
    """
    members = steering_dict[:, cluster_mask]          # (M, |C_k|)
    norms = np.linalg.norm(members, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normalised = members / norms                      # each column is unit-norm
    mean_sv = normalised.mean(axis=1)
    norm_mean = np.linalg.norm(mean_sv)
    return mean_sv / norm_mean if norm_mean > 1e-12 else mean_sv


def build_cluster_representatives(
    doa_vectors: np.ndarray,
    steering_dict: np.ndarray,
    k: int = 3,
    n_iter: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run spherical k-means and return per-cluster steering vectors.

    Parameters
    ----------
    doa_vectors   : (G, 3) unit DOA vectors for the organ
    steering_dict : (M, G) complex steering dictionary
    k             : number of clusters
    n_iter        : spherical k-means iterations
    seed          : random seed

    Returns
    -------
    S       : (M, k) complex — one representative steering vector per cluster
    labels  : (G,) cluster assignments
    centroids: (k, 3) unit cluster centroids (DOA space)
    """
    labels, centroids = spherical_kmeans(doa_vectors, k=k, n_iter=n_iter, seed=seed)
    M = steering_dict.shape[0]
    S = np.zeros((M, k), dtype=complex)
    for ki in range(k):
        mask = labels == ki
        if mask.any():
            S[:, ki] = cluster_to_steering_vector(mask, steering_dict)
    return S, labels, centroids


def whiten_subspace(
    S: np.ndarray,
    method: str = "svd",
    tol: float = 1e-8,
) -> np.ndarray:
    """Orthonormalise the columns of S to obtain a whitened basis Q.

    Used to convert a set of (possibly correlated) cluster steering vectors
    into an orthonormal null-constraint basis, so that the LCMV null conditions
    are linearly independent and well-conditioned.

    Parameters
    ----------
    S      : (M, K) complex steering vector matrix
    method : "svd" (default) — retains components with singular values above
             tol × max(s); or "qr" — keeps all K columns orthonormalised
    tol    : relative tolerance for SVD rank truncation (ignored for "qr")

    Returns
    -------
    Q : (M, r) orthonormal columns spanning the column space of S
    """
    S = np.asarray(S, dtype=complex)
    if method == "svd":
        U, s, _ = np.linalg.svd(S, full_matrices=False)
        r = int(np.sum(s > s[0] * tol))
        return U[:, :max(r, 1)]
    if method == "qr":
        Q, _ = np.linalg.qr(S, mode="reduced")
        return Q
    raise ValueError(f"method must be 'svd' or 'qr', got '{method}'")


def cluster_compactness(doa_vectors: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Compute per-cluster compactness as mean angular spread (radians).

    Lower values → tighter cluster → more stable beamformer pass direction.

    Parameters
    ----------
    doa_vectors : (G, 3) unit DOA vectors
    labels      : (G,) cluster assignments
    k           : number of clusters

    Returns
    -------
    compactness : (k,) mean angular deviation within each cluster
    """
    compactness = np.zeros(k)
    for ki in range(k):
        members = doa_vectors[labels == ki]
        if len(members) < 2:
            compactness[ki] = 0.0
            continue
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 1e-12 else centroid
        cosines = np.clip(members @ centroid, -1.0, 1.0)
        compactness[ki] = float(np.arccos(cosines).mean())
    return compactness
