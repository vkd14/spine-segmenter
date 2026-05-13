# Lumbar L1–L5 Vertebral Body Segmentation & 3D Reconstruction

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)
![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-2C5F2D.svg)
![License](https://img.shields.io/badge/use-academic-blue.svg)

End-to-end ML pipeline that takes a sagittal T1 FLAIR (or T1 / T2 reconstructed)
MRI volume in and produces:

1. A **cleaned multi-label NIfTI** segmentation of L1–L5 vertebral bodies.
2. **Five smoothed STL surface meshes** (one per vertebra) in world coordinates
   that drop directly into 3D Slicer.
3. **An interactive native macOS desktop app** with a 3-pane multi-planar viewer,
   a Plotly 3D mesh viewer, slice-scrubbing sliders, and one-click batch export.

**CS-712: Image Processing — Spring 2026 · Term Project.**

---

## Pipeline in one picture

```
   ┌───────────────┐   ┌─────────────────────┐   ┌────────────────────┐
   │ Raw MRI       │   │  Stage 1            │   │  Stage 2           │
   │ (T1 FLAIR /   │ → │  3D Reconstruction  │ → │  nnU-Net Inference │
   │  T1 / T2)     │   │  (INR-SVR)          │   │  ResEncUNet Large  │
   └───────────────┘   └─────────────────────┘   └────────────────────┘
                                                          │
   ┌───────────────────────────────────────────────┐      │
   │  Stage 3 — Anatomical Post-Processing         │ ←────┘
   │   1. Sagittal-strip corridor   (±20% LR)      │
   │   2. Largest connected component (≥ 50 vox)   │
   │   3. Vertical SI ordering enforcement         │
   └───────────────────────────────────────────────┘
                       │
                       ↓
   ┌────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
   │ Cleaned NIfTI  │ → │ 5 STL meshes     │ → │ 3-pane MPR + 3D mesh │
   │ (uint8, RAS)   │   │ (marching cubes  │   │ in the desktop app   │
   │                │   │  + Taubin)       │   │                      │
   └────────────────┘   └──────────────────┘   └──────────────────────┘
```

Three deterministic, anatomically-informed rules eliminate the dominant
failure modes (mirror artefacts, disconnected fragments, ordering violations)
of a purely data-driven model, in voxel space, with no thresholds to tune.

---

## What you get per case

```
results/<case_id>/
├── segmentation.nii.gz          # cleaned multi-label volume (uint8: L1=1 .. L5=5)
├── labels.ctbl                  # Slicer color table
├── meshes/
│   ├── L1.stl                   # one STL per vertebra in LPS world coordinates
│   ├── L2.stl
│   ├── L3.stl
│   ├── L4.stl
│   ├── L5.stl
│   └── all_vertebrae.stl        # combined surface
├── raw_prediction/
│   └── <case_id>.nii.gz         # pre-postprocess nnU-Net output (QA)
└── slicer_README.md             # Slicer load instructions
```

---

## Install

```bash
git clone git@github.com:vkd14/spine-segmenter.git
cd spine-segmenter

python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

On Apple Silicon, PyTorch ships MPS support out of the box; no special wheel
needed. The first install pulls about 2 GB of dependencies (`torch`, `nnunetv2`,
`SimpleITK`, `trimesh`, `streamlit`, etc.) and takes 5–10 minutes.

### Trained model weights (required for inference)

The five trained fold checkpoints are **not in the git repository** because they
total ~11 GB. Download them separately and place the directory at the project
root so the layout looks like:

```
spine-segmenter/
└── nnUNet_results/
    └── Dataset100_SpineL1L5/
        └── nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/
            ├── dataset.json
            ├── plans.json
            ├── fold_0/checkpoint_best.pth
            ├── fold_1/checkpoint_best.pth
            ├── fold_2/checkpoint_best.pth
            ├── fold_3/checkpoint_best.pth
            └── fold_4/checkpoint_best.pth
```

> **Note for course graders:** weights are submitted alongside the source on
> Canvas as a separate archive (or on request via email — see *Contact*).

---

## Run

### Option 1 — Native macOS app (recommended)

Double-click `launch_spine_seg_app.command` from Finder. A real macOS window
opens (WKWebView via `pywebview`); no browser, real Dock entry.

In the app:

- Drag one or many `.nii` / `.nii.gz` volumes onto the uploader (up to 600 MB each).
- Pick **1-fold (default, fast)** or **5-fold ensemble (best quality)**.
- Click **Run inference**.
- Inspect outputs live: 3-pane MPR with bilinear MRI + nearest-neighbour labels,
  Plotly 3D mesh, slice sliders. Bundles are also saved to `./results/<case_id>/`.

### Option 2 — Command line

Single case:

```bash
.venv/bin/python cli.py /data/case001_T1FLAIR.nii.gz \
    --model-dir ./nnUNet_results \
    --output ./results/case001 \
    --folds 0
```

Batch folder:

```bash
.venv/bin/python cli.py /data/flair_volumes/ \
    --model-dir ./nnUNet_results \
    --output ./results
```

Useful flags: `--folds 0 1 2 3 4` (full ensemble), `--device {cuda,mps,cpu}`,
`--coord-system {LPS,RAS}`, `--smoothing 10`, `--drop-raw`,
`--sagittal-fraction 0.20`, `--min-component-voxels 50`.  Full list:
`.venv/bin/python cli.py -h`.

### Option 3 — Python API

```python
from spine_seg import run_pipeline

result = run_pipeline(
    input_nifti="case.nii.gz",
    model_root="./nnUNet_results",
    output_dir="./results/case",
    folds=(0,),
    device_pref="mps",
    coord_system="LPS",
)
print(result.label_voxel_counts)   # e.g. {1: 28864, 2: 29458, 3: 31560, 4: 29749, 5: 25266}
print(result.stl_per_label)        # {1: PosixPath('.../L1.stl'), ...}
```

---

## Run the self-tests

```bash
.venv/bin/python test_pipeline.py
```

Builds a synthetic 5-vertebra phantom with an intentional mirror artefact and a
stray fragment, runs every post-processing step, exports STLs, and verifies
voxel→world coordinates round-trip through the affine. No GPU required.

---

## Project structure

```
spine-segmenter/
├── README.md                      ← you are here
├── requirements.txt
├── .gitignore
├── launch_spine_seg_app.command   # double-click launcher (Finder)
├── desktop_app.py                 # native-window wrapper (pywebview + WKWebView)
├── app.py                         # Streamlit GUI
├── cli.py                         # CLI entry point
├── test_pipeline.py               # synthetic-phantom self-tests
├── .streamlit/
│   └── config.toml                # 600 MB upload cap, fileWatcherType etc.
├── spine_seg/                     # core package
│   ├── __init__.py
│   ├── config.py                  # Labels, colors, ModelConfig, PostProcessConfig
│   ├── inference.py               # nnUNetPredictor wrapper, MPS/CPU autodetect
│   ├── postprocess.py             # 3-step anatomical cleanup
│   ├── mesh_export.py             # Marching cubes → STL (LPS or RAS)
│   ├── slicer_export.py           # Color table + Slicer README
│   ├── viewer.py                  # MPR figure + Plotly 3D mesh figure
│   └── pipeline.py                # End-to-end orchestrator
├── presentation/                  # term-project deck (Beamer + pptx + figures)
├── pipeline_package/              # training-side reference scripts (read-only)
└── resnet_l1-l5_pipeline.pdf      # technical reference describing the trained model
```

---

## Performance

Measured on Apple M1 Max (32-core GPU) using MPS:

| Mode | Time / case | Notes |
|---|---|---|
| 1-fold (MPS, default)        | **~12 min**    | interactive use |
| 5-fold ensemble (MPS)        | ~50–60 min     | best quality |
| 1-fold CPU fallback          | ~25–35 min     | no GPU required |
| Post-processing (per case)   | < 1 s          | SciPy CC + numpy |
| Mesh extraction (5 STLs)     | ~3–4 s         | marching cubes + Taubin |

Reference RTX 5090 numbers from the published nnU-Net pipeline: ~15 s
(1 fold), ~60 s (5-fold ensemble) per case.

---

## Limitations

- **T1-FLAIR-only model.** T2 / STIR will produce poor segmentations until
  re-trained on those contrasts. The 3D-reconstruction front-end (Yu-Erh's
  INR-SVR component, see appendix slides) is the path to support arbitrary
  modalities via super-resolution to a canonical isotropic volume.
- **Standard-anatomy assumption.** The SI-ordering rule assumes a 5-vertebra
  lumbar spine. Transitional vertebrae (lumbarised S1, sacralised L5) may
  legitimately lose a label — inspect `raw_prediction/` for QA.
- **Apple-Silicon inference latency.** 5-fold ensemble is too slow for
  real-time use on M1 Max; CUDA hardware collapses it to seconds.
- **Voxel-space post-processing.** LR/SI axes are inferred from the labeled
  bounding box rather than the NIfTI affine — robust but does not handle
  arbitrarily-oblique acquisitions.

---

## Team

This is a team project for **CS-712: Image Processing (Spring 2026)**.

| Member | Role | Contribution |
|---|---|---|
| **Varun Kumar Dasoju** ([@vkd14](https://github.com/vkd14)) | *Lead — ML pipeline & app* | nnU-Net training & 5-fold CV, 3-step anatomical post-processing, STL mesh extraction (marching cubes + Taubin), 50 % of the desktop app, repo & docs |
| **Megha K** | *Application & UI* | 50 % of the desktop app — Streamlit interface, Plotly 3D viewer, slice-scrubbing sliders, multi-case batch handling, progress reporting |
| **Yu-Erh Pan** | *3D high-resolution reconstruction* | Slice-to-Volume Reconstruction via Implicit Neural Representations (INR-SVR); continuous forward model with deformation field, multi-res hash grid, volume MLP, bias correction, Monte Carlo PSF; evaluation through TotalSegmentator |

---

## Acknowledgements

- **nnU-Net v2** (Isensee et al., *Nature Methods*, 2021) — the segmentation
  framework on which the model is built.
- **TotalSegmentator** — used as a reference / evaluation segmentor in the
  3D-reconstruction component.
- **NiBabel**, **SimpleITK**, **scikit-image**, **trimesh**, **SciPy**,
  **Streamlit**, **Plotly**, **pywebview**.

---

## Contact

**Varun Kumar Dasoju** — [@vkd14](https://github.com/vkd14) ·
the.v007@icloud.com

For questions about the term-project submission, the trained model weights,
or to reproduce the experiments end-to-end, please get in touch by email.
