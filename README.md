# mcLCMV — Anatomy-guided multicluster LCMV beamforming for EHG-MRI

Code repository for the paper:

> **Disentangling uterine and bladder activity in paired EHG-MRI data using anatomy-guided multicluster beamforming**  
> M. C. Bustos-Vivas, S. Tripathy, J. Aviles Verdera, N. Alves de Castro, J. Hutter  
> MICCAI 2026

---

## Overview

The pipeline separates uterine and bladder electrical activity recorded with a synchronized 8-channel abdominal EHG array and simultaneous T1-weighted MRI. MRI-derived 3D organ segmentations provide subject-specific steering dictionaries. A windowed dual LCMV beamformer with organ-specific pass and null constraints, Bayesian covariance shrinkage, and overlap-add reconstruction produces organ-resolved time series.

```
Stage 01  scripts/01_preprocess.py    MRI surface extraction + EHG trimming
Stage 02  scripts/02_lcmv.py          Windowed dual LCMV beamformer
Stage 03  scripts/03_results_table.py Results table
```

## Installation

```bash
git clone https://github.com/CamilaBustos258/mcLCMV_EHG_MRI.git
cd mcLCMV_EHG_MRI
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

## Data

Participant data are not included in this repository. Raw EHG and MRI files must be placed in a BIDS-organised `sourcedata/` folder:

```
sourcedata/
└── sub-XX/
    └── ses-YYYYMMDD/
        ├── mri/   (T1w.nii.gz, label-uterus_mask.nii.gz,
        │           label-bladder_mask.nii.gz, label-electrodes_mask.nii.gz)
        └── ehg/   (*.vhdr, *.vmrk, *.eeg)
```

Point the pipeline to your sourcedata folder by setting the environment variable before running any script:

```bash
export MCLCMV_SOURCEDATA=/path/to/your/sourcedata
```

If the variable is not set, the pipeline defaults to `data/sourcedata/` inside the project tree.

## Usage

```bash
# Preprocess one session
python scripts/01_preprocess.py --subject sub-06 --session ses-20250828

# Run LCMV beamformer
python scripts/02_lcmv.py --subject sub-06 --session ses-20250828

# Print results table (all processed sessions)
python scripts/03_results_table.py
```

Add `--all` to process every session registered in `config/orientation_registry.json`.

## Tests

```bash
pytest
```

All unit tests run without any data files.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bustosvivas2026mclcmv,
  title     = {Disentangling uterine and bladder activity in paired {EHG-MRI}
               data using anatomy-guided multicluster beamforming},
  author    = {Bustos-Vivas, Maria Camila and Tripathy, Smiti and
               Aviles Verdera, Jordina and Alves de Castro, Nyvenn and
               Hutter, Jana},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```
