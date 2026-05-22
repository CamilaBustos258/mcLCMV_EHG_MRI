"""MRI I/O — NIfTI read/write, orientation correction, and DICOM helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from mclcmv.config import ORIENTATION_MAP


def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI file and return the array and the image object.

    Returns
    -------
    data : np.ndarray — squeezed voxel array (trailing singleton dims removed).
    img  : nibabel.Nifti1Image — full image with affine / header.
    """
    img = nib.load(str(path))
    return np.squeeze(img.get_fdata()), img


def apply_orientation(volume: np.ndarray, orientation_case: str) -> np.ndarray:
    """Transpose and flip a 3-D volume to a consistent anatomical frame.

    The per-session ``orientation_case`` key selects the axis permutation and
    flip directions defined in ``ORIENTATION_MAP``.
    """
    if orientation_case not in ORIENTATION_MAP:
        raise ValueError(
            f"Unknown orientation_case '{orientation_case}'. "
            f"Valid keys: {list(ORIENTATION_MAP)}"
        )
    orient = ORIENTATION_MAP[orientation_case]
    out = np.transpose(volume, orient["transpose"])
    for axis in orient["flip"]:
        if axis == "X":
            out = out[::-1, :, :]
        elif axis == "Y":
            out = out[:, ::-1, :]
        elif axis == "Z":
            out = out[:, :, ::-1]
        else:
            raise ValueError(f"Unknown flip axis '{axis}'")
    return out


def load_session_volumes(
    session_mri_dir: str | Path,
    sex: str,
    orientation_case: str,
) -> dict[str, Any]:
    """Load and orient all NIfTI volumes for one session.

    Parameters
    ----------
    session_mri_dir : path to ``data/sourcedata/sub-XX/ses-YYYYMMDD/mri/``
    sex             : "F" or "M" — determines whether a uterus mask is expected
    orientation_case: key in ORIENTATION_MAP

    Returns
    -------
    dict with keys:
        T1_data, bladder_mask, electrodes_mask  – oriented np.ndarray (all sessions)
        uterus_mask                              – oriented np.ndarray or None (F only)
        voxel_sizes                              – np.ndarray [3] in mm, reordered
    """
    mri_dir = Path(session_mri_dir)

    t1_img = nib.load(str(mri_dir / "T1w.nii.gz"))
    T1_data = np.squeeze(t1_img.get_fdata())
    bladder_mask = np.squeeze(nib.load(str(mri_dir / "label-bladder_mask.nii.gz")).get_fdata())
    electrodes_mask = np.squeeze(
        nib.load(str(mri_dir / "label-electrodes_mask.nii.gz")).get_fdata()
    )

    uterus_mask: np.ndarray | None = None
    if sex == "F":
        uterus_path = mri_dir / "label-uterus_mask.nii.gz"
        if uterus_path.exists():
            uterus_mask = np.squeeze(nib.load(str(uterus_path)).get_fdata())
        else:
            warnings.warn(
                f"Uterus mask not found for female session at {mri_dir}. "
                "Session will be treated as mask-missing and skipped in uterus analysis.",
                stacklevel=2,
            )

    # Apply orientation to every volume
    T1_data = apply_orientation(T1_data, orientation_case)
    bladder_mask = apply_orientation(bladder_mask, orientation_case)
    electrodes_mask = apply_orientation(electrodes_mask, orientation_case)
    if uterus_mask is not None:
        uterus_mask = apply_orientation(uterus_mask, orientation_case)

    # Reorder voxel sizes to match the transposed axes
    voxel_sizes_orig = nib.affines.voxel_sizes(t1_img.affine)
    transpose_axes = ORIENTATION_MAP[orientation_case]["transpose"]
    voxel_sizes = np.array([voxel_sizes_orig[i] for i in transpose_axes])

    return {
        "T1_data": T1_data,
        "bladder_mask": bladder_mask,
        "electrodes_mask": electrodes_mask,
        "uterus_mask": uterus_mask,
        "voxel_sizes": voxel_sizes,
    }


def load_epi_mmode(
    dicom_dir: str | Path,
    echo: int = 1,
    frame_idx: int = 4,
    col_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read a multi-echo EPI DICOM folder and return both M-mode and 2-D frames.

    The folder is assumed to follow the project convention:
    ``[echo1_t1, echo2_t1, echo3_t1, echo1_t2, ...]``
    i.e. 3 files per volume ordered as echo 1 → 2 → 3.

    Each DICOM file contains ``NumberOfFrames`` sagittal slices stored as a
    3-D pixel array of shape ``(n_frames, rows, cols)``.  One slice
    (``frame_idx``) is extracted from every echo-1 volume.  The full 2-D image
    for that slice is returned as ``frames_2d`` for animation; a single pixel
    column (``col_idx``, default = centre) is returned as ``mmode`` for the
    spatial × time M-mode strip.

    Parameters
    ----------
    dicom_dir : folder containing ``*.dcm`` files
    echo      : which echo to extract (1-based; default 1 = shortest TE)
    frame_idx : which sagittal frame inside each multi-frame file (0-based)
    col_idx   : pixel column for the M-mode strip; defaults to centre column

    Returns
    -------
    mmode     : (n_volumes, n_rows) float32 — intensity vs time (M-mode strip)
    frames_2d : (n_volumes, n_rows, n_cols) float32 — full 2-D image per volume
    tr_sec    : float — volume repetition time in seconds
    """
    import pydicom

    dicom_dir = Path(dicom_dir)
    dcm_files = sorted(
        (p for p in dicom_dir.glob("*.dcm") if not p.name.startswith("._")),
        key=lambda p: int(p.stem),
    )

    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    # 3 echoes per volume — confirmed from DICOM headers
    n_echoes = 3
    echo_files = dcm_files[echo - 1 :: n_echoes]
    n_volumes = len(echo_files)

    if n_volumes == 0:
        raise ValueError(f"No files found for echo {echo} in {dicom_dir}")

    # Read first file to determine dimensions
    sample = pydicom.dcmread(str(echo_files[0]))
    arr0 = sample.pixel_array  # (n_frames, rows, cols)
    _, n_rows, n_cols = arr0.shape
    if col_idx is None:
        col_idx = n_cols // 2

    mmode     = np.zeros((n_volumes, n_rows), dtype=np.float32)
    frames_2d = np.zeros((n_volumes, n_rows, n_cols), dtype=np.float32)

    mmode[0]     = arr0[frame_idx, :, col_idx].astype(np.float32)
    frames_2d[0] = arr0[frame_idx].astype(np.float32)

    for i, fpath in enumerate(echo_files[1:], start=1):
        ds = pydicom.dcmread(str(fpath))
        sl = ds.pixel_array[frame_idx].astype(np.float32)
        frames_2d[i] = sl
        mmode[i]     = sl[:, col_idx]

    # TR: confirmed from R255 marker spacing (850 samples @ 500 Hz = 1.7 s)
    tr_sec = 850.0 / 500.0
    try:
        if hasattr(sample, "TriggerTime"):
            ds_next = pydicom.dcmread(str(echo_files[1]))
            tr_sec = (float(ds_next.TriggerTime) - float(sample.TriggerTime)) / 1000.0
    except Exception:
        pass

    return mmode, frames_2d, tr_sec
