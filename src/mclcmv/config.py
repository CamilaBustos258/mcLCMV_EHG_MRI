"""Global configuration constants and paths for the mcLCMV pipeline.

Follows a BIDS-inspired directory structure under data/.
Set the environment variable MCLCMV_SOURCEDATA to point to an external
sourcedata folder when the raw data lives outside the project tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Project paths ──────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_ROOT: Path = PROJECT_ROOT / "data"

# Source data location — resolved with a three-level cascade:
#   1. Environment variable MCLCMV_SOURCEDATA  (highest priority)
#   2. data_config.yaml at the project root    (gitignored — local path stays private)
#   3. data/sourcedata/ inside the repo        (fallback for self-contained setups)
#
# To configure:
#   cp data_config.yaml.example data_config.yaml   # then edit sourcedata_root
# or:
#   export MCLCMV_SOURCEDATA=/path/to/your/sourcedata

import os as _os


def _resolve_sourcedata() -> Path:
    """Return the resolved SOURCEDATA path (env var > data_config.yaml > default)."""
    # 1. Environment variable
    if "MCLCMV_SOURCEDATA" in _os.environ:
        return Path(_os.environ["MCLCMV_SOURCEDATA"])
    # 2. data_config.yaml (gitignored, local only)
    _cfg = PROJECT_ROOT / "data_config.yaml"
    if _cfg.exists():
        for _line in _cfg.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line.startswith("sourcedata_root") and ":" in _line:
                _, _, _val = _line.partition(":")
                return Path(_val.strip()).expanduser().resolve()
    # 3. Default: data/sourcedata/ inside the repo
    return DATA_ROOT / "sourcedata"


SOURCEDATA: Path = _resolve_sourcedata()

DERIVATIVES:    Path = DATA_ROOT / "derivatives"
DERIV_SYNCED:   Path = DERIVATIVES / "synced"    # session_data.npz outputs
DERIV_ANALYSIS: Path = DERIVATIVES / "analysis"  # LCMV results

ORIENTATION_REGISTRY_PATH: Path = PROJECT_ROOT / "config" / "orientation_registry.json"

# ── MRI orientation map ────────────────────────────────────────────────────
# Each case specifies the axis permutation and flip needed to bring the NIfTI
# volume into a consistent Right-Anterior-Superior (RAS) anatomical frame.
# Keys match the orientation_case values in orientation_registry.json.
ORIENTATION_MAP: dict[str, dict[str, Any]] = {
    "default":           {"transpose": (0, 1, 2), "flip": []},
    "T1_(0,2,1)_flipZ":  {"transpose": (0, 2, 1), "flip": ["Z"]},
    "T1_(2,0,1)_flipZ":  {"transpose": (2, 0, 1), "flip": ["Z"]},
    "T1_(2,1,0)_flipZ":  {"transpose": (2, 1, 0), "flip": ["Z"]},
    "T1_(1,2,0)_flipZ":  {"transpose": (1, 2, 0), "flip": ["Z"]},
}


def load_orientation_registry() -> dict[str, dict[str, Any]]:
    """Load and return the per-session orientation registry."""
    with open(ORIENTATION_REGISTRY_PATH) as f:
        return json.load(f)


def load_active_sessions() -> list[tuple[str, str]]:
    """Return all (subject, session) pairs not marked ``"skip": true``.

    Sessions can be excluded from batch processing by adding ``"skip": true``
    to their registry entry (e.g. while a segmentation is pending).  Single-
    session runs with ``--subject``/``--session`` always proceed regardless of
    this flag.
    """
    registry = load_orientation_registry()
    return [
        (sub, ses)
        for sub, sessions in registry.items()
        for ses, meta in sessions.items()
        if not meta.get("skip", False)
    ]


# ── Frequency bands (Hz) ───────────────────────────────────────────────────
UTERUS_BAND_HZ:  tuple[float, float] = (0.005, 0.050)   # [5, 50] mHz
BLADDER_BAND_HZ: tuple[float, float] = (0.025, 0.041)   # [25, 41] mHz

# ── Beamformer defaults ────────────────────────────────────────────────────
SHRINKAGE_BETA:    float = 0.10    # Bayesian diagonal shrinkage factor β
DIAG_LOAD_FACTOR:  float = 1e-3   # additional diagonal loading (× trace/M)
N_CLUSTERS:        int   = 3      # K_U = K_B = 3 spherical k-means clusters
N_NULL_DIRECTIONS: int   = 2      # null directions retained per opposing organ
N_KMEANS_ITER:     int   = 60     # spherical k-means iterations

# ── Overlap-add window ─────────────────────────────────────────────────────
WINDOW_SIZE_SEC: float = 120.0    # Hann-tapered window length (s)
STEP_SIZE_SEC:   float = 30.0     # step → 75 % overlap (Hann COLA)

# ── Electrode extraction (DBSCAN) ─────────────────────────────────────────
DBSCAN_EPS_CM:         float = 0.47
DBSCAN_MIN_SAMPLES:    int   = 5
EXPECTED_N_ELECTRODES: int   = 8

# ── Steering vector exponent ───────────────────────────────────────────────
# p ∈ [1, 2]; quasi-static bio-electric model: a_i(r) ∝ ‖r_i − r‖^{−p}
STEERING_EXPONENT: float = 1.5

# ── Surface subsampling cap ────────────────────────────────────────────────
MAX_SURFACE_VERTS: int = 3000
