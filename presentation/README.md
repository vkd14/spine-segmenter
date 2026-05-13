# Term Project Presentation

Beamer presentation for the lumbar L1-L5 segmentation project.

## Files

```
presentation/
├── presentation.tex       # the main Beamer document (10 sections, 23 frames)
├── _make_figures.py       # regenerates the figures from the smoke-test outputs
├── README.md              # this file
└── figures/
    ├── mri_only.png            # raw 3-pane MPR (no segmentation)
    ├── mpr_overlay.png         # 3-pane MPR with colored seg overlay
    ├── postprocess_steps.png   # 4-panel before/after-each-step view
    ├── mesh_3d.png             # rendered 3D surface (all 5 vertebrae)
    └── voxel_counts.png        # per-vertebra voxel-count bar chart
```

## How to compile

### Easiest — Overleaf (online, no install)

1. Go to <https://overleaf.com>, **New Project → Upload Project**, and zip up
   this `presentation/` folder before uploading.
2. Overleaf auto-detects `presentation.tex` as the main file. Hit **Recompile**.
3. The Metropolis theme is pre-installed there; document compiles in ~10 s.

### Local — MacTeX or BasicTeX

Install once:
```bash
brew install --cask basictex          # ~100 MB; lighter than full MacTeX (~5 GB)
sudo tlmgr update --self
sudo tlmgr install latexmk metropolis pgfopts beamertheme-metropolis
```

Then build:
```bash
cd presentation
latexmk -pdf presentation.tex          # or: pdflatex presentation.tex   (twice for TOC)
open presentation.pdf
```

The document falls back to the built-in `Madrid` theme if Metropolis isn't
available, so it'll still compile on a minimal install — just less pretty.

## Regenerating the figures

The figures in `figures/` were rendered from the actual smoke-test pipeline
outputs (case `542_20240612_Lumbar_Spine_Sagittal_T1_FLAIR_s3`). To regenerate
them after running new cases:

```bash
.venv/bin/python presentation/_make_figures.py
```

The script reads from `results/smoke_test_542/` by default — edit the constants
at the top of `_make_figures.py` to point at a different case.
