"""Visualization helpers for the mcLCMV pipeline.

All functions return the figure object so callers can save or further
customise before displaying.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_overlays(
    data: np.ndarray,
    fs: float,
    y_uterus: np.ndarray,
    y_bladder: np.ndarray,
    y_bladder_clean: np.ndarray | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    max_channels: int = 4,
    title: str = "Time-series overlay",
) -> plt.Figure:
    """Plot raw EHG channels alongside uterus and bladder beamformer outputs.

    Parameters
    ----------
    data            : (M, N) raw multi-channel EHG
    fs              : sampling frequency in Hz
    y_uterus        : (N,) uterus beamformer output
    y_bladder       : (N,) bladder beamformer output (pre-leakage subtraction)
    y_bladder_clean : (N,) bladder output after leakage subtraction (optional)
    t_start, t_end  : time window to display in seconds (None = full recording)
    max_channels    : maximum number of raw channels to show (default 4)
    title           : figure title

    Returns
    -------
    fig : matplotlib Figure
    """
    M, N = data.shape
    t = np.arange(N) / fs

    t_start = t_start if t_start is not None else 0.0
    t_end   = t_end   if t_end   is not None else t[-1]
    idx = (t >= t_start) & (t <= t_end)

    n_ch = min(max_channels, M)
    n_rows = n_ch + (3 if y_bladder_clean is not None else 2)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.1 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(n_ch):
        axes[i].plot(t[idx], data[i, idx], color="0.4", lw=0.9)
        axes[i].set_ylabel(f"Ch {i + 1}")
        axes[i].grid(True, alpha=0.2)

    k = n_ch
    axes[k].plot(t[idx], y_uterus[idx], color="#1f77b4", lw=1.2)
    axes[k].set_ylabel("Uterus")
    axes[k].grid(True, alpha=0.2)

    axes[k + 1].plot(t[idx], y_bladder[idx], color="#ff7f0e", lw=1.2)
    axes[k + 1].set_ylabel("Bladder")
    axes[k + 1].grid(True, alpha=0.2)

    if y_bladder_clean is not None:
        axes[k + 2].plot(t[idx], y_bladder_clean[idx], color="#2ca02c", lw=1.2)
        axes[k + 2].set_ylabel("Bladder\n(clean)")
        axes[k + 2].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, y=0.998)
    plt.tight_layout()
    return fig


def plot_covariance_eigenvalues(
    diagnostics: list[dict],
    which_window: int = 0,
    title: str = "Covariance eigenspectrum",
) -> plt.Figure:
    """Semilog plot of the sorted covariance eigenvalues for one window.

    Parameters
    ----------
    diagnostics  : list returned by ``run_windowed_lcmv``
    which_window : index of the window to inspect (default 0)
    title        : figure title

    Returns
    -------
    fig : matplotlib Figure
    """
    evals = np.abs(np.array(diagnostics[which_window].get("R_eig", [])))
    if evals.size == 0:
        raise ValueError(
            "diagnostics entries do not contain 'R_eig'. "
            "Re-run run_windowed_lcmv with store_covariance=True."
        )
    evals_sorted = np.sort(evals)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(evals_sorted, "o-", lw=1.1)
    ax.set_xlabel("Eigenvalue rank (descending)")
    ax.set_ylabel("|Eigenvalue|")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{title} — window {which_window}")
    plt.tight_layout()
    return fig


def plot_array_response_over_vertices(
    w: np.ndarray,
    steering_dict: np.ndarray,
    organ_vertices: np.ndarray,
    title: str = "Array response over organ surface",
    marker_size: int = 20,
) -> plt.Figure:
    """3-D scatter of beamformer gain magnitude over an organ surface.

    Parameters
    ----------
    w             : (M,) complex beamformer weight vector
    steering_dict : (M, G) complex steering dictionary for the organ
    organ_vertices: (G, 3) vertex coordinates in cm
    title         : figure title
    marker_size   : scatter marker size

    Returns
    -------
    fig : matplotlib Figure
    """
    gains = np.abs(steering_dict.T @ w.conj())        # (G,)  = |D^T w*|
    gains /= gains.max() + 1e-12

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        organ_vertices[:, 0],
        organ_vertices[:, 1],
        organ_vertices[:, 2],
        c=gains,
        cmap="viridis",
        s=marker_size,
    )
    fig.colorbar(sc, ax=ax, label="Normalised gain")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_alpha_over_time(
    diagnostics: list[dict],
    fs: float,
    step_size_sec: float,
    title: str = "Leakage coefficient α over time",
) -> plt.Figure:
    """Plot the per-window leakage subtraction coefficient α.

    Parameters
    ----------
    diagnostics    : list returned by the beamformer pipeline, each entry
                     expected to have key "alpha_leakage"
    fs             : sampling frequency in Hz (used only to compute time axis)
    step_size_sec  : step size in seconds between windows
    title          : figure title

    Returns
    -------
    fig : matplotlib Figure
    """
    alphas = [d.get("alpha_leakage", np.nan) for d in diagnostics]
    times = np.arange(len(alphas)) * step_size_sec

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, alphas, "o-", lw=1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("α")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    plt.tight_layout()
    return fig
