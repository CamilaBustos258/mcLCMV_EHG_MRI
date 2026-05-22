"""Uterus–bladder separation metrics.

Implements the evaluation metrics from Section 2.2 and Table 1 of the paper:

- Morlet CWT scalograms and band-power time series P_U[n], P_B[n]
- Separation ratio ρ[n] = P_U[n] / (P_B[n] + ε)
- Leakage diagnostics: |α|, Δr (correlation change), energy ratio |y_B_clean|/|y_B|
- Geometric metrics: separation angle, cluster compactness

Reference: Bustos-Vivas et al., 2025.
"""

from __future__ import annotations

import numpy as np
import pywt


def uterus_bladder_separation_angle(
    uterus_centroid: np.ndarray,
    bladder_centroid: np.ndarray,
    array_center: np.ndarray,
) -> float:
    """Compute the angle (degrees) between uterus and bladder DOA directions.

    Parameters
    ----------
    uterus_centroid  : (3,) centroid of uterus surface in cm
    bladder_centroid : (3,) centroid of bladder surface in cm
    array_center     : (3,) electrode array geometric centre in cm

    Returns
    -------
    angle_deg : float — separation angle in degrees
    """
    d_u = uterus_centroid - array_center
    d_b = bladder_centroid - array_center
    d_u /= np.linalg.norm(d_u) + 1e-12
    d_b /= np.linalg.norm(d_b) + 1e-12
    cos_angle = np.clip(np.dot(d_u, d_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def morlet_band_power(
    signal: np.ndarray,
    fs: float,
    freq_band: tuple[float, float],
    n_freqs: int = 20,
) -> np.ndarray:
    """Compute instantaneous band power using a complex Morlet CWT.

    P[n] = Σ_{f ∈ band} |W(a_f, n)|²

    Parameters
    ----------
    signal    : (N,) time series
    fs        : sampling frequency in Hz
    freq_band : (f_low, f_high) frequency band in Hz
    n_freqs   : number of frequency bins within the band

    Returns
    -------
    band_power : (N,) instantaneous band power
    """
    freqs = np.linspace(freq_band[0], freq_band[1], n_freqs)
    scales = pywt.frequency2scale("cmor1.5-1.0", freqs / fs)
    coeffs, _ = pywt.cwt(signal, scales, "cmor1.5-1.0", sampling_period=1.0 / fs)
    return np.sum(np.abs(coeffs) ** 2, axis=0)


def separation_ratio(
    power_uterus: np.ndarray,
    power_bladder: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the uterus-to-bladder energy ratio ρ[n] = P_U / (P_B + ε).

    Parameters
    ----------
    power_uterus  : (N,) instantaneous uterus band power
    power_bladder : (N,) instantaneous bladder band power
    eps           : small constant to avoid division by zero

    Returns
    -------
    rho : (N,) separation ratio
    """
    return power_uterus / (power_bladder + eps)


def wavelet_scalogram(
    signal: np.ndarray,
    fs: float,
    freq_band: tuple[float, float],
    wavelet: str = "cmor1.5-1.0",
    n_scales: int = 256,
    smooth_time_sec: float = 0.0,
    smooth_scale_idx: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a smoothed CWT scalogram over a frequency band.

    Returns the full time–frequency power map S(f, n) = |W(a_f, n)|², with
    optional Gaussian smoothing in the time and scale dimensions.  Use
    ``morlet_band_power`` when only the band-summed power time series is needed.

    Parameters
    ----------
    signal          : (N,) time series
    fs              : sampling frequency in Hz
    freq_band       : (f_low, f_high) frequency range in Hz
    wavelet         : PyWavelets continuous wavelet name (default "cmor1.5-1.0")
    n_scales        : number of log-spaced scales across the band (default 256)
    smooth_time_sec : Gaussian smoothing σ in seconds along time axis (0 = off)
    smooth_scale_idx: Gaussian smoothing σ in scale-index units along scale axis
                      (0 = off)

    Returns
    -------
    t     : (N,) time vector in seconds
    freqs : (n_kept,) frequency axis in Hz (ascending)
    Sxx   : (n_kept, N) power scalogram
    """
    from scipy.ndimage import gaussian_filter

    dt = 1.0 / fs
    wave = pywt.ContinuousWavelet(wavelet)

    base_scales = np.logspace(1.0, 4.5, num=n_scales)
    freqs_all = pywt.scale2frequency(wave, base_scales) / dt
    keep = (freqs_all >= freq_band[0]) & (freqs_all <= freq_band[1])
    if not np.any(keep):
        raise ValueError(
            f"No scales map into band {freq_band}. "
            "Try adjusting n_scales or the wavelet parameters."
        )
    scales = base_scales[keep]
    freqs = freqs_all[keep]
    # sort ascending in frequency
    order = np.argsort(freqs)
    scales, freqs = scales[order], freqs[order]

    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
    Sxx = np.abs(coeffs) ** 2   # (n_kept, N)

    if smooth_time_sec > 0 or smooth_scale_idx > 0:
        sigma_t = smooth_time_sec * fs
        Sxx = gaussian_filter(Sxx, sigma=(smooth_scale_idx, sigma_t))

    t = np.arange(len(signal)) * dt
    return t, freqs, Sxx


def leakage_diagnostics(
    y_uterus: np.ndarray,
    y_bladder_raw: np.ndarray,
    y_bladder_clean: np.ndarray,
    alpha: float,
) -> dict:
    """Compute leakage diagnostics matching Table 1 of the paper.

    Parameters
    ----------
    y_uterus       : (N,) uterus signal
    y_bladder_raw  : (N,) bladder signal before leakage subtraction
    y_bladder_clean: (N,) bladder signal after leakage subtraction
    alpha          : scalar leakage coefficient

    Returns
    -------
    dict with keys:
        alpha          : |α| leakage coefficient
        delta_r        : Δr = r_after − r_before (correlation change; should be < 0)
        energy_ratio   : |y_B_clean| / |y_B_raw| (should be < 1)
    """
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

    r_before = corr(y_bladder_raw, y_uterus)
    r_after = corr(y_bladder_clean, y_uterus)
    energy_ratio = float(
        np.linalg.norm(y_bladder_clean) / (np.linalg.norm(y_bladder_raw) + 1e-12)
    )

    p_raw = float(np.mean(y_bladder_raw ** 2))
    p_clean = float(np.mean(y_bladder_clean ** 2))
    relative_power_drop = (p_raw - p_clean) / (p_raw + 1e-12)

    return {
        "alpha": abs(alpha),
        "delta_r": r_after - r_before,
        "energy_ratio": energy_ratio,
        "relative_power_drop": relative_power_drop,
    }
