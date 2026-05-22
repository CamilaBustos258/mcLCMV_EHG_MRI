"""EHG preprocessing — common-average reference, bandpass filtering, MRI window trimming."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import iirfilter, sosfiltfilt


def clip_amplitudes(
    data: np.ndarray,
    clip_factor: float = 10.0,
) -> np.ndarray:
    """Per-channel amplitude clipping using robust MAD-based thresholds.

    Clips each channel to ±clip_factor × MAD, where MAD is the median
    absolute deviation (a robust estimator of spread unaffected by the very
    outliers we want to remove).  Clipping preserves signal continuity
    — no gaps, no filter-edge artefacts — while capping large transients
    before they reach sosfiltfilt.  At 0.005 Hz the filter impulse response
    spans ~200 s, so even a single unclipped spike would ring for minutes.

    Parameters
    ----------
    data        : (n_channels, n_samples) — trimmed, unfiltered EHG
    clip_factor : symmetric clip threshold in units of MAD (default 10)

    Returns
    -------
    clipped : (n_channels, n_samples) — same shape, values bounded
    """
    out = data.copy()
    for i, ch in enumerate(data):
        mad = np.median(np.abs(ch - np.median(ch))) * 1.4826  # consistent σ estimator
        if mad < 1e-12:
            continue  # flat channel — leave as-is
        out[i] = np.clip(ch, -clip_factor * mad, clip_factor * mad)
    return out


def detect_bad_channels(
    data: np.ndarray,
    mad_threshold: float = 1e-4,
) -> np.ndarray:
    """Identify flat or disconnected channels.

    A channel whose MAD is below ``mad_threshold`` carries no useful signal
    (electrode off skin, broken wire, amplifier saturation at a fixed value).
    Excluding such channels from CAR prevents them from biasing the reference mean.

    Parameters
    ----------
    data          : (n_channels, n_samples)
    mad_threshold : channels with MAD < this value are flagged as bad

    Returns
    -------
    good : (n_channels,) boolean — True means the channel is usable
    """
    good = np.zeros(data.shape[0], dtype=bool)
    for i, ch in enumerate(data):
        mad = np.median(np.abs(ch - np.median(ch))) * 1.4826
        good[i] = mad >= mad_threshold
    return good


def compute_window_quality(
    data: np.ndarray,
    fs: float,
    window_size_sec: float,
    step_size_sec: float,
) -> np.ndarray:
    """Continuous quality weight in (0, 1] for every sliding window.

    For each window, computes the per-channel RMS and compares it to each
    channel's median RMS (baseline level).  The excess above baseline is
    summarised as a single weight via:

        excess  = max over channels of max(0, rms / median_rms − 1)
        quality = exp(−excess)

    Clean windows (rms ≈ median everywhere) → weight ≈ 1.
    Noisy windows (any channel has rms >> median) → weight → 0.
    No window is ever dropped — weights are used for a weighted mean.

    Parameters
    ----------
    data            : (n_channels, n_samples) — trimmed, clipped, unfiltered
    fs              : sampling frequency in Hz
    window_size_sec : window length (must match analysis parameters)
    step_size_sec   : step size  (must match analysis parameters)

    Returns
    -------
    quality : (n_windows,) float in (0, 1]
    """
    win = int(window_size_sec * fs)
    step = int(step_size_sec * fs)
    n_samples = data.shape[1]
    n_windows = max(0, (n_samples - win) // step + 1)

    if n_windows == 0:
        return np.ones(0)

    # Per-channel baseline RMS from short (10 s) sliding windows
    ref_win = max(1, int(10 * fs))
    ref_step = max(1, ref_win // 4)
    n_ref = (n_samples - ref_win) // ref_step + 1
    baseline_rms = np.zeros(data.shape[0])
    for i, ch in enumerate(data):
        rms_vals = [
            np.sqrt(np.mean(ch[k * ref_step : k * ref_step + ref_win] ** 2))
            for k in range(n_ref)
        ]
        baseline_rms[i] = np.median(rms_vals) if rms_vals else 1.0
    baseline_rms = np.where(baseline_rms < 1e-12, 1.0, baseline_rms)

    quality = np.zeros(n_windows)
    for w in range(n_windows):
        seg = data[:, w * step : w * step + win]
        rms = np.sqrt(np.mean(seg ** 2, axis=1))           # (n_channels,)
        excess = np.max(np.maximum(0.0, rms / baseline_rms - 1.0))
        quality[w] = np.exp(-excess)

    return quality


def apply_car(
    data: np.ndarray,
    good_channels: np.ndarray | None = None,
) -> np.ndarray:
    """Common-Average Reference subtraction.

    Removes the spatial mean across channels at every time point, which
    suppresses shared-mode artefacts (gradient residuals, breathing baseline)
    without attenuating spatially specific sources.

    Parameters
    ----------
    data          : (n_channels, n_samples)
    good_channels : optional boolean mask (n_channels,); if provided, only
                    good channels contribute to the mean. All channels
                    (including bad ones) have the reference subtracted.

    Returns
    -------
    data_car : (n_channels, n_samples) — mean-subtracted
    """
    if good_channels is not None and good_channels.any():
        ref = data[good_channels].mean(axis=0, keepdims=True)
    else:
        ref = data.mean(axis=0, keepdims=True)
    return data - ref


def design_bandpass(
    fs: float,
    low: float = 0.005,
    high: float = 0.05,
    order: int = 1,
) -> np.ndarray:
    """Design an IIR Butterworth bandpass filter (SOS form).

    Default passband (0.005–0.05 Hz) covers uterine peristalsis (20 s–200 s periods).

    Returns
    -------
    sos : second-order sections array suitable for ``sosfiltfilt``
    """
    return iirfilter(
        N=order,
        Wn=[low, high],
        btype="band",
        ftype="butter",
        fs=fs,
        output="sos",
    )


def apply_bandpass(
    data: np.ndarray,
    sos: np.ndarray,
    padlen: int | None = None,
) -> np.ndarray:
    """Apply a zero-phase SOS filter to every channel.

    Parameters
    ----------
    data   : (n_channels, n_samples)
    sos    : second-order sections from ``design_bandpass``
    padlen : sosfiltfilt padding length in samples.  Larger values reduce
             edge transients at the cost of more computation.  Pass
             ``int(100 * fs)`` for 100 s of synthetic reflection padding
             when filtering short trimmed recordings.  None uses scipy's
             default (≈ 3 filter time constants — usually too short at
             0.005 Hz).

    Returns
    -------
    filtered : (n_channels, n_samples)
    """
    kw: dict = {} if padlen is None else {"padlen": padlen}
    return np.array([sosfiltfilt(sos, ch, **kw) for ch in data])


def rational_resample(
    data: np.ndarray,
    fs_in: float,
    fs_out_target: float,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Downsample or upsample data using polyphase rational resampling.

    Computes the best integer up/down ratio to approximate ``fs_out_target``
    and applies ``scipy.signal.resample_poly``.  Avoids the spectral artefacts
    of simple decimation while keeping filter complexity bounded.

    Parameters
    ----------
    data          : (n_channels, n_samples) or (n_samples,)
    fs_in         : input sampling frequency in Hz
    fs_out_target : desired output sampling frequency in Hz

    Returns
    -------
    data_resampled : same shape as input except along the last axis
    fs_out         : actual output sampling frequency (may differ slightly
                     from ``fs_out_target`` due to rational approximation)
    (up, down)     : the integer up/down factors used
    """
    from fractions import Fraction
    from scipy.signal import resample_poly

    frac = Fraction(fs_out_target / fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    data_resampled = resample_poly(data, up, down, axis=-1)
    fs_out = fs_in * up / down
    return data_resampled, fs_out, (up, down)


def find_mri_trim_indices(markers_df: pd.DataFrame) -> tuple[int, int]:
    """Return (start_sample, end_sample) of the MRI acquisition window.

    Same marker logic as ``trim_to_mri_window`` but returns sample indices
    instead of sliced data, so callers can filter the full recording first
    and trim afterwards.

    Raises
    ------
    ValueError if no MRI trigger markers are found.
    """
    for code in ("R255", "R128"):
        rows = markers_df[markers_df["marker_code"] == code]
        if len(rows) >= 2:
            return int(rows["sample_index"].iloc[0]), int(rows["sample_index"].iloc[-1])

    available = markers_df["marker_code"].unique().tolist()
    raise ValueError(
        f"No MRI trigger markers (R255 / R128) found. "
        f"Available codes in .vmrk: {available}"
    )


def trim_to_mri_window(
    data: np.ndarray,
    markers_df: pd.DataFrame,
    fs: float,  # noqa: ARG001 — kept for API symmetry
) -> np.ndarray:
    """Trim EHG data to the MRI recording window.

    The first and last MRI trigger markers (R255 or R128) mark the start and
    end of the simultaneous MRI acquisition.  Samples outside this window
    contain only ambient noise (no MRI gradient artefacts but also no
    synchronised physiological signal).

    Parameters
    ----------
    data       : (n_channels, n_samples) — full recording
    markers_df : output of ``io.ehg.read_markers``
    fs         : sampling frequency (kept for API symmetry, not used)

    Returns
    -------
    trimmed : (n_channels, n_mri_samples)

    Raises
    ------
    ValueError if no MRI trigger markers are found.
    """
    start, end = find_mri_trim_indices(markers_df)
    return data[:, start:end]
