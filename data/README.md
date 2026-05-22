# data/

This folder is **gitignored** — never commit raw or processed data.

```
data/
├── sourcedata/          # raw inputs: NIfTI MRI + BrainVision EHG per subject
│   ├── sub-02/
│   │   └── ses-YYYYMMDD/
│   │       ├── mri/
│   │       │   ├── T1w.nii.gz
│   │       │   ├── label-uterus_mask.nii.gz
│   │       │   ├── label-bladder_mask.nii.gz
│   │       │   └── label-electrodes_mask.nii.gz
│   │       └── ehg/
│   │           ├── sub-02_ses-YYYYMMDD.vhdr
│   │           ├── sub-02_ses-YYYYMMDD.vmrk
│   │           └── sub-02_ses-YYYYMMDD.eeg
│   └── ...
└── derivatives/         # pipeline outputs: steering dicts, beamformer results
    └── sub-02/
        └── ses-YYYYMMDD/
            ├── session_data.npz   # preprocessed EHG + surfaces + electrode coords
            ├── steering/          # D_U, D_B, S_U, S_B arrays
            ├── beamformer/        # y_uterus, y_bladder, diagnostics
            └── meta.json
```

Subject naming follows BIDS convention: `sub-XX` (two-digit zero-padded IDs).
