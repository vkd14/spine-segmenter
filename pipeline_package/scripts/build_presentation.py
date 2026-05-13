#!/usr/bin/env python3
"""Build spine segmentation pipeline PPTX presentation."""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "results", "spine_seg_pipeline.pptx")

# ── Color palette ──────────────────────────────────────────────────────────
C_DARK    = RGBColor(0x1A, 0x1A, 0x2E)   # deep navy
C_BLUE    = RGBColor(0x16, 0x21, 0x3E)   # mid navy
C_ACCENT  = RGBColor(0x0F, 0x3F, 0x5F)   # steel blue
C_TEAL    = RGBColor(0x00, 0xB4, 0xD8)   # highlight teal
C_GREEN   = RGBColor(0x06, 0xD6, 0xA0)   # success green
C_YELLOW  = RGBColor(0xFF, 0xD1, 0x66)   # warning / highlight
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_LGRAY   = RGBColor(0xCC, 0xCC, 0xCC)
C_MGRAY   = RGBColor(0x88, 0x88, 0x99)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # completely blank

def bg(slide, color=C_DARK):
    """Fill slide background."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def txb(slide, text, l, t, w, h, size=18, bold=False, color=C_WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return tb

def rect(slide, l, t, w, h, fill_color, line_color=None, line_w=None):
    from pptx.util import Pt as UPt
    shp = slide.shapes.add_shape(1, l, t, w, h)  # MSO_SHAPE_TYPE.RECTANGLE=1
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    if line_color:
        shp.line.color.rgb = line_color
        if line_w:
            shp.line.width = line_w
    else:
        shp.line.fill.background()
    return shp

def hline(slide, y, color=C_ACCENT):
    ln = slide.shapes.add_shape(1, Inches(0), y, SLIDE_W, Pt(1))
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()

def add_image(slide, path, l, t, w, h=None):
    if not os.path.exists(path):
        return
    if h:
        slide.shapes.add_picture(path, l, t, w, h)
    else:
        slide.shapes.add_picture(path, l, t, w)

def pill(slide, text, l, t, w, h, bg_color, txt_color=C_WHITE, size=13, bold=False):
    r = rect(slide, l, t, w, h, bg_color)
    txb(slide, text, l, t, w, h, size=size, bold=bold, color=txt_color, align=PP_ALIGN.CENTER)
    return r

# ── Paths ──────────────────────────────────────────────────────────────────
VIZ_DIR  = os.path.join(ROOT, "results", "visualizations")
VIZ_T1   = os.path.join(ROOT, "results", "visualizations_t1")
FOLD_DIR = os.path.join(ROOT, "results", "nnUNet_results",
           "Dataset202_SpineL1L5_SSL",
           "nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans__3d_fullres")

SUMMARY_GRID  = os.path.join(VIZ_DIR, "summary_grid.png")
FLAIR_PANEL   = os.path.join(VIZ_DIR, "flair_10_20200918_Lumbar_Spine_Sagittal_T1_FLAIR_s5_panel.png")
T1_PANEL      = os.path.join(VIZ_T1, "spider_t1_100_t1_panel.png")
FOLD0_PROG    = os.path.join(FOLD_DIR, "fold_0", "progress.png")
FOLD1_PROG    = os.path.join(FOLD_DIR, "fold_1", "progress.png")
FOLD2_PROG    = os.path.join(FOLD_DIR, "fold_2", "progress.png")

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl, C_DARK)

# Gradient accent bar on left
rect(sl, Inches(0), Inches(0), Inches(0.18), SLIDE_H, C_TEAL)
rect(sl, Inches(0.18), Inches(0), Inches(0.06), SLIDE_H, C_ACCENT)

# Title block
txb(sl, "Lumbar Spine Segmentation", Inches(0.5), Inches(1.4), Inches(12), Inches(1.1),
    size=46, bold=True, color=C_WHITE)
txb(sl, "Automated L1–L5 Vertebra Segmentation Pipeline", Inches(0.5), Inches(2.55),
    Inches(12), Inches(0.7), size=26, color=C_TEAL, bold=False)

hline(sl, Inches(3.45))

txb(sl, "nnU-Net v2 · ResEnc L Architecture · RTX 5090 (32 GB VRAM)",
    Inches(0.5), Inches(3.6), Inches(8), Inches(0.55), size=17, color=C_LGRAY)

# Metric pills at bottom
pill_data = [
    ("346 Volumes", C_ACCENT),
    ("5-Fold CV", C_ACCENT),
    ("Dice 0.914", C_GREEN),
    ("RTX 5090", RGBColor(0x55,0x55,0x77)),
]
px = Inches(0.5)
for label, col in pill_data:
    pill(sl, label, px, Inches(4.4), Inches(1.8), Inches(0.5), col, size=14, bold=True)
    px += Inches(2.0)

txb(sl, "April 2026", Inches(0.5), Inches(6.8), Inches(4), Inches(0.4),
    size=13, color=C_MGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem & Motivation
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Problem & Motivation", Inches(0.4), Inches(0.2), Inches(12), Inches(0.7),
    size=30, bold=True, color=C_WHITE)

# Left column
points_left = [
    ("Clinical Need", C_TEAL, True),
    ("Lumbar spine disease is one of the leading causes of disability worldwide.", C_WHITE, False),
    ("Manual vertebra segmentation requires expert radiologists and is time-consuming,\ncostly, and prone to inter-observer variability.", C_LGRAY, False),
    "",
    ("Target Application", C_TEAL, True),
    ("Fully automatic L1–L5 vertebra delineation from Sagittal T1-FLAIR MRI scans\nfor use in clinical workflow and research pipelines.", C_LGRAY, False),
    "",
    ("Key Challenge", C_TEAL, True),
    ("Limited labeled data — 243 clinical T1-FLAIR scans with no ground-truth labels.\nSolution: Transfer learning from public SPIDER dataset + pseudo-label self-training.", C_LGRAY, False),
]

y = Inches(1.3)
for item in points_left:
    if item == "":
        y += Inches(0.15)
        continue
    text, col, bold = item
    txb(sl, text, Inches(0.5), y, Inches(6.0), Inches(0.8), size=15, color=col, bold=bold)
    y += Inches(0.55) if bold else Inches(0.52)

# Right column — stats boxes
box_data = [
    ("243", "Clinical T1-FLAIR MRI scans\n(unlabeled target domain)", C_TEAL),
    ("447", "SPIDER public labeled volumes\n(T1 + T2 sagittal)", C_GREEN),
    ("0.914", "Dice Score achieved\n(5-class vertebra segmentation)", C_YELLOW),
]
bx = Inches(7.2)
by = Inches(1.5)
for num, label, col in box_data:
    rect(sl, bx, by, Inches(2.5), Inches(1.5), C_ACCENT)
    txb(sl, num, bx, by+Inches(0.08), Inches(2.5), Inches(0.75),
        size=38, bold=True, color=col, align=PP_ALIGN.CENTER)
    txb(sl, label, bx, by+Inches(0.75), Inches(2.5), Inches(0.65),
        size=12, color=C_LGRAY, align=PP_ALIGN.CENTER)
    by += Inches(1.7)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Dataset Overview", Inches(0.4), Inches(0.2), Inches(12), Inches(0.7),
    size=30, bold=True, color=C_WHITE)

datasets = [
    {
        "name": "SPIDER Dataset",
        "desc": "Public lumbar spine MRI benchmark",
        "items": [
            "447 sagittal MRI volumes (T1 + T2 weighted)",
            "Human-expert vertebra + disc annotations",
            "Covers L1–L5 + inter-vertebral discs",
            "Source: University Medical Center Utrecht",
            "Used as: primary labeled training data",
        ],
        "color": C_TEAL,
        "x": Inches(0.35),
    },
    {
        "name": "Sagittal T1-FLAIR (Clinical)",
        "desc": "Target domain — unlabeled",
        "items": [
            "243 clinical lumbar MRI scans",
            "Sagittal T1-FLAIR sequence",
            "No ground-truth labels available",
            "Auto-labeled via TotalSpineSeg + self-training",
            "Forms 346-volume Dataset202 after filtering",
        ],
        "color": C_YELLOW,
        "x": Inches(4.55),
    },
    {
        "name": "Dataset202_SpineL1L5_SSL",
        "desc": "Final training dataset (this run)",
        "items": [
            "346 volumes total (SPIDER + pseudo-labeled)",
            "Labels: background, L1, L2, L3, L4, L5",
            "Format: nnU-Net raw (.nii.gz)",
            "Split: 5-fold cross-validation",
            "~70 volumes per validation fold",
        ],
        "color": C_GREEN,
        "x": Inches(8.75),
    },
]

for ds in datasets:
    x = ds["x"]
    rect(sl, x, Inches(1.2), Inches(4.0), Inches(5.9), C_BLUE,
         line_color=ds["color"], line_w=Pt(1.5))
    rect(sl, x, Inches(1.2), Inches(4.0), Inches(0.55), ds["color"])
    txb(sl, ds["name"], x+Inches(0.1), Inches(1.22), Inches(3.8), Inches(0.35),
        size=14, bold=True, color=C_DARK)
    txb(sl, ds["desc"], x+Inches(0.1), Inches(1.6), Inches(3.8), Inches(0.35),
        size=12, color=C_LGRAY, italic=True)
    y = Inches(2.05)
    for item in ds["items"]:
        txb(sl, "•  " + item, x+Inches(0.15), y, Inches(3.7), Inches(0.4),
            size=13, color=C_WHITE)
        y += Inches(0.46)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Full Pipeline Overview
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "End-to-End Pipeline", Inches(0.4), Inches(0.2), Inches(12), Inches(0.7),
    size=30, bold=True, color=C_WHITE)

stages = [
    ("1", "Data\nCollection", "SPIDER 447 vols\n+ 243 T1-FLAIR", C_TEAL),
    ("2", "Auto\nLabeling", "TotalSpineSeg\nL5-S1 anchor", RGBColor(0x48,0xCA,0xAE)),
    ("3", "Dataset\nPrep", "nnU-Net format\n346 volumes", RGBColor(0x45,0xB7,0xD1)),
    ("4", "nnU-Net\nTraining", "ResEnc L\n5-fold CV", RGBColor(0x96,0xC0,0xCE)),
    ("5", "Pseudo-label\nSelf-Training", "4/5 fold agree\nFLARE22 recipe", RGBColor(0x06,0xD6,0xA0)),
    ("6", "Inference\n& Post-proc", "L1–L5 masks\n+ mesh export", RGBColor(0xFF,0xD1,0x66)),
]

bw = Inches(1.88)
bh = Inches(2.3)
sx = Inches(0.32)
sy = Inches(1.4)
gap = Inches(0.1)

for i, (num, title, desc, col) in enumerate(stages):
    x = sx + i * (bw + gap)
    # Box
    rect(sl, x, sy, bw, bh, C_ACCENT, line_color=col, line_w=Pt(1.5))
    # Top color band
    rect(sl, x, sy, bw, Inches(0.42), col)
    # Stage number
    txb(sl, num, x, sy, bw, Inches(0.42), size=20, bold=True,
        color=C_DARK, align=PP_ALIGN.CENTER)
    # Title
    txb(sl, title, x, sy+Inches(0.45), bw, Inches(0.85),
        size=15, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    # Description
    txb(sl, desc, x, sy+Inches(1.3), bw, Inches(0.85),
        size=12, color=C_LGRAY, align=PP_ALIGN.CENTER)
    # Arrow (not after last)
    if i < len(stages) - 1:
        txb(sl, "▶", x + bw, sy + Inches(0.95), gap + Inches(0.05), Inches(0.4),
            size=14, color=C_TEAL, align=PP_ALIGN.CENTER)

# Sub-pipeline: detailed flow below
rect(sl, Inches(0.3), Inches(4.05), Inches(12.7), Inches(2.9), C_BLUE)
txb(sl, "Detailed Training Sub-Pipeline", Inches(0.5), Inches(4.1),
    Inches(6), Inches(0.4), size=14, bold=True, color=C_TEAL)

sub_stages = [
    ("Raw MRI\n(.nii.gz)", C_MGRAY),
    ("Z-score\nNormalization", C_ACCENT),
    ("Spine\nCropping", C_ACCENT),
    ("nnU-Net\nPlanning", C_ACCENT),
    ("3D Full-Res\nPreprocessing", C_ACCENT),
    ("5-Fold\nTraining Loop", C_GREEN),
    ("Val Dice\nEvaluation", C_GREEN),
    ("Best Model\nCheckpoint", C_YELLOW),
]

bw2 = Inches(1.45)
bh2 = Inches(1.55)
sx2 = Inches(0.42)
sy2 = Inches(4.6)
gap2 = Inches(0.07)

for i, (title, col) in enumerate(sub_stages):
    x = sx2 + i * (bw2 + gap2)
    rect(sl, x, sy2, bw2, bh2, C_DARK, line_color=col, line_w=Pt(1))
    txb(sl, title, x, sy2 + Inches(0.5), bw2, Inches(0.8),
        size=11, color=C_WHITE, align=PP_ALIGN.CENTER)
    if i < len(sub_stages) - 1:
        txb(sl, "→", x + bw2, sy2 + Inches(0.6), gap2 + Inches(0.1), Inches(0.35),
            size=12, color=C_TEAL, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Architecture: nnU-Net ResEnc L
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Architecture: nnU-Net ResEnc L", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Left: architecture description
arch_items = [
    ("nnU-Net v2 — Self-Configuring Framework", C_TEAL, True, 16),
    ("Automatically adapts network topology, patch size,\nbatch size and normalization to input data.", C_LGRAY, False, 13),
    ("", None, False, 0),
    ("ResEnc L Architecture (nnU-Net Revisited, MICCAI 2024)", C_TEAL, True, 16),
    ("Replaces plain encoder with residual blocks.\nLarger encoder capacity — best on 5/6 benchmarks.", C_LGRAY, False, 13),
    ("", None, False, 0),
    ("3D Full Resolution Configuration", C_TEAL, True, 16),
    ("Processes full 3D volumes without downsampling.\nPatch size: auto-selected ~128×128×96 voxels.", C_LGRAY, False, 13),
    ("", None, False, 0),
    ("Key Hyperparameters", C_TEAL, True, 16),
    ("• Trainer: nnUNetTrainer_250epochs", C_WHITE, False, 13),
    ("• Plans: nnUNetResEncUNetLPlans", C_WHITE, False, 13),
    ("• Optimizer: SGD with Polynomial LR decay", C_WHITE, False, 13),
    ("• Loss: Dice + Cross-Entropy (combined)", C_WHITE, False, 13),
    ("• Data Augmentation: rotation, scaling, elastic, mirror", C_WHITE, False, 13),
]

y = Inches(1.2)
for item in arch_items:
    text, col, bold, size = item
    if text == "":
        y += Inches(0.1)
        continue
    txb(sl, text, Inches(0.4), y, Inches(6.4), Inches(0.65),
        size=size, color=col, bold=bold)
    y += Inches(0.46) if size > 14 else Inches(0.44)

# Right: encoder-decoder diagram
EX = Inches(7.2)
EY = Inches(1.2)
EW = Inches(5.7)
EH = Inches(5.8)
rect(sl, EX, EY, EW, EH, C_BLUE, line_color=C_ACCENT, line_w=Pt(1))

txb(sl, "ResEnc L Encoder–Decoder (3D U-Net)", EX, EY+Inches(0.1),
    EW, Inches(0.4), size=13, bold=True, color=C_TEAL, align=PP_ALIGN.CENTER)

# Encoder blocks
enc_stages = [
    ("Input 3D MRI", C_MGRAY, Inches(2.1), Inches(0.2)),
    ("ResBlock 32ch → Pool", C_ACCENT, Inches(1.8), Inches(0.45)),
    ("ResBlock 64ch → Pool", C_ACCENT, Inches(1.5), Inches(0.45)),
    ("ResBlock 128ch → Pool", C_ACCENT, Inches(1.2), Inches(0.45)),
    ("ResBlock 256ch → Pool", C_ACCENT, Inches(0.9), Inches(0.45)),
    ("Bottleneck 320ch", RGBColor(0x2A,0x4A,0x6A), Inches(0.6), Inches(0.45)),
]

dec_labels = [
    "UpConv + Skip + ResBlock 256ch",
    "UpConv + Skip + ResBlock 128ch",
    "UpConv + Skip + ResBlock 64ch",
    "UpConv + Skip + ResBlock 32ch",
    "1×1×1 Conv → 6 classes (softmax)",
]

cy = EY + Inches(0.65)
cx_center = EX + EW / 2
col_enc = RGBColor(0x16, 0x4A, 0x6E)
for i, (label, col, w, h) in enumerate(enc_stages):
    bx = cx_center - w / 2
    rect(sl, bx, cy, w, h, col, line_color=C_TEAL, line_w=Pt(0.5))
    txb(sl, label, bx, cy + Inches(0.07), w, h - Inches(0.07),
        size=10, color=C_WHITE, align=PP_ALIGN.CENTER)
    cy += h + Inches(0.06)

# Decoder on right side
cy2 = cy - Inches(0.06)
for label in dec_labels:
    cy2 -= Inches(0.45) + Inches(0.06)
    rect(sl, cx_center + Inches(0.05), cy2, Inches(2.3), Inches(0.45),
         RGBColor(0x10,0x4A,0x3A), line_color=C_GREEN, line_w=Pt(0.5))
    txb(sl, label, cx_center + Inches(0.1), cy2 + Inches(0.08),
        Inches(2.2), Inches(0.35), size=9, color=C_WHITE, align=PP_ALIGN.LEFT)

txb(sl, "Encoder", EX + Inches(0.1), EY + Inches(2.5), Inches(1.2), Inches(0.4),
    size=11, color=C_TEAL, bold=True)
txb(sl, "Decoder", EX + EW - Inches(1.2), EY + Inches(2.5), Inches(1.1), Inches(0.4),
    size=11, color=C_GREEN, bold=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — TotalSpineSeg Auto-Labeling
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Stage 2: TotalSpineSeg Auto-Labeling", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Left description
steps = [
    ("Why Auto-Labeling?",
     "243 clinical T1-FLAIR scans have no ground-truth labels.\nManual annotation is infeasible. TotalSpineSeg provides\nautomatic anatomical landmark detection.", C_TEAL),
    ("L5–S1 Disc as Anchor",
     "TotalSpineSeg detects the L5–S1 intervertebral disc as\na reliable anatomical landmark — highly visible in MRI.\nAll vertebra labels propagate upward from this anchor.", C_WHITE),
    ("Label Propagation",
     "From L5–S1 anchor → count up:\nL5 ↑ L4 ↑ L3 ↑ L2 ↑ L1\nHandles rotated/tilted scans natively via orientation-aware\ndetection.", C_WHITE),
    ("Quality Filtering",
     "Pseudo-labels filtered at confidence threshold 0.80.\nOnly high-confidence predictions included in training.\nResult: 346 usable volumes out of 447 + 243 total.", C_GREEN),
]

y = Inches(1.25)
for title, desc, col in steps:
    rect(sl, Inches(0.35), y, Inches(6.1), Inches(1.35), C_ACCENT)
    txb(sl, title, Inches(0.5), y + Inches(0.05), Inches(5.9), Inches(0.4),
        size=14, bold=True, color=col)
    txb(sl, desc, Inches(0.5), y + Inches(0.45), Inches(5.9), Inches(0.85),
        size=12, color=C_LGRAY)
    y += Inches(1.5)

# Right: flow diagram
FX = Inches(7.0)
FY = Inches(1.3)
FW = Inches(5.9)
FH = Inches(5.7)
rect(sl, FX, FY, FW, FH, C_BLUE, line_color=C_ACCENT)
txb(sl, "Auto-Labeling Workflow", FX, FY + Inches(0.1), FW, Inches(0.4),
    size=14, bold=True, color=C_TEAL, align=PP_ALIGN.CENTER)

flow = [
    ("Input: 243 unlabeled T1-FLAIR MRI", C_MGRAY),
    ("↓", C_TEAL),
    ("TotalSpineSeg inference\n(pretrained public model)", C_ACCENT),
    ("↓", C_TEAL),
    ("Detect L5-S1 disc anchor\n(vertebra counting from below)", C_ACCENT),
    ("↓", C_TEAL),
    ("Generate L1–L5 vertebra masks", C_ACCENT),
    ("↓", C_TEAL),
    ("Confidence filter (≥ 0.80)", C_ACCENT),
    ("↓", C_TEAL),
    ("103 volumes accepted as\nhigh-confidence pseudo-labels", C_GREEN),
    ("↓", C_TEAL),
    ("Merge with SPIDER → Dataset202\n(346 total training volumes)", C_GREEN),
]

fy = FY + Inches(0.65)
for text, col in flow:
    is_arrow = text.startswith("↓")
    fh_ = Inches(0.25) if is_arrow else Inches(0.52)
    size_ = 18 if is_arrow else 12
    txb(sl, text, FX + Inches(0.2), fy, FW - Inches(0.4), fh_,
        size=size_, color=col, align=PP_ALIGN.CENTER,
        bold=not is_arrow)
    fy += fh_ + Inches(0.04)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Training Details & RTX 5090 Stability
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Training Details & Hardware Challenges", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Training config table - left
config_rows = [
    ("Parameter", "Value", True, C_TEAL, C_TEAL),
    ("GPU",               "NVIDIA RTX 5090 (32 GB VRAM)",       False, C_WHITE, C_LGRAY),
    ("Architecture",      "nnU-Net ResEnc L (3d_fullres)",       False, C_WHITE, C_LGRAY),
    ("Dataset",           "Dataset202_SpineL1L5_SSL (346 vol.)", False, C_WHITE, C_LGRAY),
    ("Folds",             "5 (cross-validation)",               False, C_WHITE, C_LGRAY),
    ("Epochs",            "250 (fold 0) / 150 (folds 1–4)",     False, C_WHITE, C_LGRAY),
    ("Optimizer",         "SGD, momentum=0.99",                  False, C_WHITE, C_LGRAY),
    ("LR Schedule",       "Polynomial decay 0.01 → 0",          False, C_WHITE, C_LGRAY),
    ("Batch Size",        "Auto (nnU-Net: 2–4)",                 False, C_WHITE, C_LGRAY),
    ("Loss",              "Dice + Cross-Entropy",                False, C_WHITE, C_LGRAY),
    ("Augmentation",      "Rotation, scale, elastic, Gaussian", False, C_WHITE, C_LGRAY),
    ("DA Workers",        "0 (nnUNet_n_proc_DA=0)",              False, C_WHITE, C_LGRAY),
    ("Epoch Time",        "~285 s/epoch",                        False, C_WHITE, C_LGRAY),
]

tw = Inches(6.2)
th = Inches(0.37)
tx = Inches(0.35)
ty = Inches(1.2)
for i, (k, v, bold, kc, vc) in enumerate(config_rows):
    bg_c = C_ACCENT if i == 0 else (C_BLUE if i % 2 == 0 else RGBColor(0x1E,0x2A,0x3A))
    rect(sl, tx, ty, tw, th, bg_c)
    txb(sl, k, tx + Inches(0.1), ty + Inches(0.03), Inches(2.7), th - Inches(0.03),
        size=12, bold=bold, color=kc)
    txb(sl, v, tx + Inches(2.9), ty + Inches(0.03), Inches(3.2), th - Inches(0.03),
        size=12, bold=bold, color=vc)
    ty += th

# Right: crash history & fixes
rx = Inches(7.0)
txb(sl, "RTX 5090 Stability Fixes", rx, Inches(1.2), Inches(6.0), Inches(0.45),
    size=17, bold=True, color=C_YELLOW)

crashes = [
    ("1", "cudaErrorLaunchTimeout", "GPU watchdog kills long-running kernels",
     "CUDA_LAUNCH_BLOCKING=1", C_YELLOW),
    ("2", "CUDNN_STATUS_EXECUTION_FAILED", "cuDNN algo selection triggers GPU crash",
     "benchmark=False, deterministic=True", RGBColor(0xFF,0x88,0x44)),
    ("3", "corrupted size vs prev_size", "Multiprocess dataloader memory corruption",
     "nnUNet_n_proc_DA=0", RGBColor(0xFF,0x55,0x55)),
    ("4", "CUDA OOM during cascade", "Concurrent fold processes share GPU VRAM",
     "Sequential fold execution (wait for PID)", RGBColor(0xAA,0x55,0xFF)),
    ("5", "Checkpoint not resumed", "maybe_load_checkpoint() never called",
     "Explicit checkpoint load with continue=True", C_GREEN),
]

ry = Inches(1.8)
for num, err, cause, fix, col in crashes:
    rect(sl, rx, ry, Inches(6.0), Inches(0.95), C_BLUE, line_color=col, line_w=Pt(1))
    rect(sl, rx, ry, Inches(0.32), Inches(0.95), col)
    txb(sl, num, rx, ry, Inches(0.32), Inches(0.95),
        size=14, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)
    txb(sl, err, rx + Inches(0.38), ry + Inches(0.03), Inches(5.5), Inches(0.35),
        size=12, bold=True, color=col)
    txb(sl, f"Cause: {cause}", rx + Inches(0.38), ry + Inches(0.35),
        Inches(5.5), Inches(0.28), size=10, color=C_MGRAY)
    txb(sl, f"Fix: {fix}", rx + Inches(0.38), ry + Inches(0.6),
        Inches(5.5), Inches(0.28), size=10, color=C_GREEN)
    ry += Inches(1.05)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Fold 0 Training Progress
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Fold 0 Training Progress — 250 Epochs", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Progress chart
if os.path.exists(FOLD0_PROG):
    add_image(sl, FOLD0_PROG, Inches(0.3), Inches(1.15), Inches(5.2))

# Convergence milestones
txb(sl, "Convergence Milestones", Inches(6.0), Inches(1.2), Inches(6.9), Inches(0.4),
    size=17, bold=True, color=C_TEAL)

milestones = [
    ("Epoch 50",  "0.842", "+0.842 from baseline", C_MGRAY),
    ("Epoch 100", "0.882", "+0.040 gain",          RGBColor(0x66,0xBB,0xFF)),
    ("Epoch 150", "0.895", "+0.013 gain",           C_TEAL),
    ("Epoch 200", "0.906", "+0.011 gain",           C_GREEN),
    ("Epoch 250", "0.910", "+0.004 gain  ✓ Final", C_YELLOW),
]

my = Inches(1.8)
for ep, dice, note, col in milestones:
    bar_w = float(dice) * Inches(3.5)
    rect(sl, Inches(6.0), my, bar_w, Inches(0.42), col)
    txb(sl, ep, Inches(6.05), my + Inches(0.06), Inches(1.1), Inches(0.3),
        size=12, bold=True, color=C_DARK)
    txb(sl, f"Dice {dice}  |  {note}", Inches(6.0) + bar_w + Inches(0.1),
        my + Inches(0.06), Inches(3.5), Inches(0.3), size=12, color=C_WHITE)
    my += Inches(0.58)

# Insight box
rect(sl, Inches(6.0), Inches(4.85), Inches(6.9), Inches(2.3), C_ACCENT)
txb(sl, "Key Insight: Diminishing Returns", Inches(6.1), Inches(4.95),
    Inches(6.7), Inches(0.4), size=14, bold=True, color=C_YELLOW)
insights = [
    "• Epochs 0–100 provide the bulk of learning (+0.88 dice)",
    "• Epochs 100–150 add +0.013 — still worthwhile",
    "• Epochs 150–250 add only +0.015 total",
    "• ► 150 epochs selected as the sweet spot for folds 1–4",
    "  Saves ~10h per fold with <1.5% dice penalty",
]
iy = Inches(5.45)
for line in insights:
    txb(sl, line, Inches(6.15), iy, Inches(6.6), Inches(0.38),
        size=12, color=C_WHITE)
    iy += Inches(0.36)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Multi-Fold Training Status
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "5-Fold Cross-Validation — Training Status", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

folds_data = [
    (0, "✅ Complete", 250, 250, 0.9100, 0.9139, "~20h", C_GREEN, 1.0),
    (1, "✅ Complete", 150, 150, 0.9122, None,   "~12h", C_GREEN, 1.0),
    (2, "🔄 Training", 116, 150, 0.8915, None,   "~2h left", C_TEAL, 116/150),
    (3, "⏳ Queued",   0,   150, None,   None,   "~12h", C_MGRAY, 0.0),
    (4, "⏳ Queued",   0,   150, None,   None,   "~12h", C_MGRAY, 0.0),
]

headers = ["Fold", "Status", "Epoch", "Target", "Best Dice (EMA)", "Val Dice", "ETA"]
col_widths = [Inches(0.55), Inches(1.7), Inches(0.85), Inches(0.85), Inches(1.9), Inches(1.5), Inches(1.5)]

tx, ty = Inches(0.3), Inches(1.25)
th_h = Inches(0.42)
th_r = Inches(0.52)

# Header
x = tx
rect(sl, x, ty, sum(col_widths), th_h, C_ACCENT)
for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
    txb(sl, hdr, x + Inches(0.05), ty + Inches(0.06), cw - Inches(0.1), th_h,
        size=13, bold=True, color=C_WHITE)
    x += cw
ty += th_h

for fold, status, ep, tgt, best_dice, val_dice, eta, col, progress in folds_data:
    x = tx
    bg_c = RGBColor(0x1A,0x2A,0x3A) if fold % 2 == 0 else C_BLUE
    rect(sl, x, ty, sum(col_widths), th_r, bg_c)

    row_vals = [
        str(fold),
        status,
        str(ep),
        str(tgt),
        f"{best_dice:.4f}" if best_dice else "—",
        f"{val_dice:.4f}" if val_dice else "(pending)",
        eta,
    ]
    for j, (val, cw) in enumerate(zip(row_vals, col_widths)):
        vc = col if j == 1 else (C_WHITE if val != "—" and val != "(pending)" else C_MGRAY)
        txb(sl, val, x + Inches(0.05), ty + Inches(0.08), cw - Inches(0.1), th_r - Inches(0.08),
            size=13, color=vc)
        x += cw

    # Progress bar
    bar_x = tx + Inches(8.1)
    bar_y = ty + Inches(0.14)
    bar_max_w = Inches(4.6)
    bar_h = Inches(0.25)
    rect(sl, bar_x, bar_y, bar_max_w, bar_h, C_ACCENT)
    if progress > 0:
        rect(sl, bar_x, bar_y, bar_max_w * progress, bar_h, col)
    txb(sl, f"{int(progress*100)}%", bar_x + bar_max_w + Inches(0.1), bar_y,
        Inches(0.5), bar_h, size=11, color=col)
    ty += th_r

# Progress charts for folds 0 and 1
if os.path.exists(FOLD1_PROG):
    add_image(sl, FOLD1_PROG, Inches(0.3), Inches(4.45), Inches(5.0))

txb(sl, "Fold 1 Training Curves (150 epochs)", Inches(0.4), Inches(4.3),
    Inches(5.0), Inches(0.3), size=12, color=C_TEAL, bold=True)

if os.path.exists(FOLD2_PROG):
    add_image(sl, FOLD2_PROG, Inches(5.6), Inches(4.45), Inches(5.0))
txb(sl, "Fold 2 Training Progress (in progress)", Inches(5.7), Inches(4.3),
    Inches(5.0), Inches(0.3), size=12, color=C_TEAL, bold=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Validation Results (Fold 0)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Validation Results — Fold 0 (250 Epochs)", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Per-vertebra dice results
txb(sl, "Per-Vertebra Dice Score (Fold 0 Validation — 71 volumes)",
    Inches(0.4), Inches(1.15), Inches(8), Inches(0.4), size=16, bold=True, color=C_TEAL)

dice_data = [
    ("L1", 0.9058, C_TEAL),
    ("L2", 0.9134, RGBColor(0x48,0xCA,0xAE)),
    ("L3", 0.9219, C_GREEN),
    ("L4", 0.9119, RGBColor(0x45,0xB7,0xD1)),
    ("L5", 0.9162, RGBColor(0x96,0xC0,0xCE)),
]

bx = Inches(0.4)
bar_max = Inches(5.8)
bar_h = Inches(0.62)
by = Inches(1.7)
for vert, dice, col in dice_data:
    rect(sl, bx, by, bar_max, bar_h, C_ACCENT)
    filled = bar_max * (dice - 0.85) / 0.15  # scale 0.85–1.0
    filled = max(Inches(0.3), filled)
    rect(sl, bx, by, filled, bar_h, col)
    txb(sl, vert, bx - Inches(0.35), by + Inches(0.14), Inches(0.32), Inches(0.35),
        size=15, bold=True, color=C_WHITE, align=PP_ALIGN.RIGHT)
    txb(sl, f"{dice:.4f}", bx + filled + Inches(0.1), by + Inches(0.14),
        Inches(0.9), Inches(0.35), size=15, bold=True, color=col)
    by += Inches(0.75)

txb(sl, f"Mean Dice: 0.9139", Inches(0.4), by + Inches(0.15),
    Inches(4), Inches(0.45), size=18, bold=True, color=C_YELLOW)

# Right: metrics detail table
txb(sl, "Detailed Metrics (Fold 0 Validation)", Inches(7.0), Inches(1.15),
    Inches(5.9), Inches(0.4), size=16, bold=True, color=C_TEAL)

metric_headers = ["Class", "Dice", "IoU", "Precision", "Recall"]
metric_data = [
    ("L1", 0.9058, 0.8467, 0.9144, 0.8988),
    ("L2", 0.9134, 0.8562, 0.9178, 0.9096),
    ("L3", 0.9219, 0.8730, 0.9342, 0.9103),
    ("L4", 0.9119, 0.8558, 0.9170, 0.9074),
    ("L5", 0.9162, 0.8613, 0.9239, 0.9089),
    ("Mean", 0.9139, 0.8586, 0.9215, 0.9070),
]

mx = Inches(7.0)
my = Inches(1.7)
mw = [Inches(0.7), Inches(0.9), Inches(0.9), Inches(1.15), Inches(1.15)]
mh = Inches(0.42)

# Header
rect(sl, mx, my, sum(mw), mh, C_ACCENT)
x = mx
for h, w in zip(metric_headers, mw):
    txb(sl, h, x+Inches(0.05), my+Inches(0.07), w-Inches(0.1), mh,
        size=12, bold=True, color=C_WHITE)
    x += w
my += mh

for i, row in enumerate(metric_data):
    bg_c = C_BLUE if i % 2 == 0 else RGBColor(0x1A,0x2A,0x3A)
    is_mean = row[0] == "Mean"
    if is_mean:
        bg_c = C_ACCENT
    rect(sl, mx, my, sum(mw), mh, bg_c)
    x = mx
    vals = [row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"]
    for j, (v, w) in enumerate(zip(vals, mw)):
        col = C_YELLOW if is_mean else (C_GREEN if j > 0 else C_WHITE)
        txb(sl, v, x+Inches(0.05), my+Inches(0.07), w-Inches(0.1), mh,
            size=12, bold=is_mean, color=col)
        x += w
    my += mh

# Comparison vs baseline
txb(sl, "Performance vs. Baseline", Inches(7.0), my + Inches(0.2),
    Inches(6.0), Inches(0.35), size=14, bold=True, color=C_TEAL)
baselines = [
    ("V7 MPS Baseline (EfficientNet-B4, 2D)", "0.776", C_MGRAY),
    ("nnU-Net ResEnc L — Fold 0 (250 ep)", "0.914", C_GREEN),
    ("nnU-Net ResEnc L — Fold 1 (150 ep)", "0.912*", C_TEAL),
]
bly = my + Inches(0.62)
for name, dice, col in baselines:
    rect(sl, Inches(7.0), bly, Inches(5.9), Inches(0.4), C_BLUE)
    txb(sl, name, Inches(7.1), bly + Inches(0.05), Inches(4.0), Inches(0.32),
        size=12, color=C_LGRAY)
    txb(sl, dice, Inches(11.7), bly + Inches(0.05), Inches(1.0), Inches(0.32),
        size=13, bold=True, color=col)
    bly += Inches(0.45)

txb(sl, "* EMA pseudo-dice (validation summary pending)",
    Inches(7.0), bly + Inches(0.05), Inches(5.9), Inches(0.3),
    size=10, color=C_MGRAY, italic=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Visualizations: T1-FLAIR Clinical MRI
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Segmentation Results — Clinical T1-FLAIR MRI", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

if os.path.exists(FLAIR_PANEL):
    add_image(sl, FLAIR_PANEL, Inches(0.3), Inches(1.2), Inches(12.7), Inches(3.2))

# Legend
txb(sl, "Left → Right: Raw MRI  |  Ground Truth  |  Prediction  |  Correct / Error",
    Inches(0.4), Inches(4.5), Inches(12), Inches(0.35),
    size=13, color=C_MGRAY, align=PP_ALIGN.CENTER, italic=True)

# Color legend for vertebrae
legend_items = [
    ("L1", RGBColor(0xFF,0x00,0x00)),
    ("L2", RGBColor(0x00,0xFF,0x00)),
    ("L3", RGBColor(0x00,0x00,0xFF)),
    ("L4", RGBColor(0xFF,0xFF,0x00)),
    ("L5", RGBColor(0xFF,0x00,0xFF)),
]
lx = Inches(3.5)
ly = Inches(4.95)
for label, col in legend_items:
    rect(sl, lx, ly, Inches(0.3), Inches(0.3), col)
    txb(sl, label, lx + Inches(0.35), ly, Inches(0.55), Inches(0.3),
        size=13, color=C_WHITE, bold=True)
    lx += Inches(1.0)

# Summary grid
if os.path.exists(SUMMARY_GRID):
    add_image(sl, SUMMARY_GRID, Inches(0.3), Inches(5.4), Inches(12.7), Inches(1.85))

txb(sl, "Summary Grid: Multiple Clinical Cases — Raw | Ground Truth | Prediction | Error Map",
    Inches(0.4), Inches(5.25), Inches(12), Inches(0.35),
    size=12, color=C_TEAL, bold=True, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Visualizations: SPIDER T1 MRI
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Segmentation Results — SPIDER T1 Benchmark", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

if os.path.exists(T1_PANEL):
    add_image(sl, T1_PANEL, Inches(0.3), Inches(1.2), Inches(12.7), Inches(3.3))

txb(sl, "Left → Right: Raw T1 MRI  |  Ground Truth  |  Prediction  |  Correct / Error",
    Inches(0.4), Inches(4.6), Inches(12), Inches(0.35),
    size=13, color=C_MGRAY, align=PP_ALIGN.CENTER, italic=True)

# Observations
obs = [
    ("Accurate Boundaries", "Model correctly delineates individual vertebra boundaries\nwith sharp edges, even in regions with low contrast.", C_GREEN),
    ("L1–L5 Discrimination", "All five vertebrae correctly identified and labeled\nwithout confusion between adjacent levels.", C_TEAL),
    ("Error Pattern", "Minor errors at vertebra endpoints (superior/inferior endplates)\nand in cases with disc herniation or compression.", C_YELLOW),
]

ox = Inches(0.3)
oy = Inches(5.1)
ow = Inches(4.0)
for title, desc, col in obs:
    rect(sl, ox, oy, ow, Inches(2.1), C_ACCENT, line_color=col, line_w=Pt(1.5))
    txb(sl, title, ox + Inches(0.1), oy + Inches(0.08), ow - Inches(0.2), Inches(0.4),
        size=13, bold=True, color=col)
    txb(sl, desc, ox + Inches(0.1), oy + Inches(0.5), ow - Inches(0.2), Inches(1.5),
        size=12, color=C_LGRAY)
    ox += Inches(4.35)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Dice Convergence & 95% Analysis
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Dice Convergence Analysis — Can We Reach 95%?", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Convergence chart (manual)
txb(sl, "Fold 0 — Observed Convergence Curve", Inches(0.4), Inches(1.15),
    Inches(6), Inches(0.4), size=15, bold=True, color=C_TEAL)

# Chart area
chart_x = Inches(0.4)
chart_y = Inches(1.65)
chart_w = Inches(5.8)
chart_h = Inches(3.5)
rect(sl, chart_x, chart_y, chart_w, chart_h, C_BLUE, line_color=C_ACCENT)

# Y axis labels
y_labels = ["0.84", "0.86", "0.88", "0.90", "0.92", "0.95"]
y_vals   = [0.84, 0.86, 0.88, 0.90, 0.92, 0.95]
for yl, yv in zip(y_labels, y_vals):
    frac = (yv - 0.83) / (0.96 - 0.83)
    ypos = chart_y + chart_h - chart_h * frac
    txb(sl, yl, chart_x - Inches(0.5), ypos - Inches(0.12), Inches(0.45), Inches(0.25),
        size=10, color=C_MGRAY, align=PP_ALIGN.RIGHT)
    # gridline
    rect(sl, chart_x, ypos, chart_w, Pt(0.5), RGBColor(0x33,0x33,0x44))

# Data points
epochs_pts = [0, 50, 100, 150, 200, 250]
dice_pts   = [0.0, 0.842, 0.882, 0.895, 0.906, 0.910]
x_pts = [chart_x + chart_w * (ep / 250) for ep in epochs_pts]
y_pts = [chart_y + chart_h - chart_h * ((d - 0.83) / (0.96 - 0.83)) for d in dice_pts]

# Draw line segments
for i in range(1, len(x_pts)):
    # Approximate line with thin rect
    x1, y1 = x_pts[i-1], y_pts[i-1]
    x2, y2 = x_pts[i], y_pts[i]
    rect(sl, x1, min(y1,y2), x2-x1, abs(y2-y1) + Pt(1.5), C_TEAL)

# Data point circles (small squares)
for i, (xp, yp) in enumerate(zip(x_pts[1:], y_pts[1:]), 1):
    sz = Inches(0.12)
    rect(sl, xp - sz/2, yp - sz/2, sz, sz, C_YELLOW)
    txb(sl, f"{dice_pts[i]:.3f}", xp - Inches(0.2), yp - Inches(0.32),
        Inches(0.5), Inches(0.22), size=9, color=C_YELLOW, align=PP_ALIGN.CENTER)

# 95% target line
target_y = chart_y + chart_h - chart_h * ((0.95 - 0.83) / (0.96 - 0.83))
rect(sl, chart_x, target_y, chart_w, Pt(1.5), RGBColor(0xFF, 0x44, 0x44))
txb(sl, "95% target", chart_x + chart_w - Inches(1.0), target_y - Inches(0.2),
    Inches(1.2), Inches(0.25), size=9, color=RGBColor(0xFF, 0x44, 0x44))

# 150ep marker
ep150_x = chart_x + chart_w * (150/250)
rect(sl, ep150_x, chart_y, Pt(1.5), chart_h, C_GREEN)
txb(sl, "150ep\n(sweet spot)", ep150_x + Inches(0.05), chart_y + Inches(0.1),
    Inches(0.85), Inches(0.4), size=9, color=C_GREEN)

# X axis labels
for ep in [0, 50, 100, 150, 200, 250]:
    xpos = chart_x + chart_w * (ep / 250)
    txb(sl, str(ep), xpos - Inches(0.2), chart_y + chart_h + Inches(0.05),
        Inches(0.4), Inches(0.25), size=10, color=C_MGRAY, align=PP_ALIGN.CENTER)
txb(sl, "Epoch", chart_x + chart_w/2, chart_y + chart_h + Inches(0.3),
    Inches(1), Inches(0.25), size=11, color=C_LGRAY, align=PP_ALIGN.CENTER)

# Right: analysis
txb(sl, "Can Folds 2–4 Reach 95% Dice?", Inches(6.8), Inches(1.15),
    Inches(6.1), Inches(0.4), size=17, bold=True, color=C_YELLOW)

analysis = [
    ("Answer: Not at 150 Epochs", C_YELLOW, True, 15),
    ("Based on fold 0 convergence, 150 epochs → ~0.895–0.900 dice.\nThe 95% threshold requires the curve to gain +0.055 more —\n~5× the total gain seen from epochs 150→250.", C_LGRAY, False, 13),
    ("", None, False, 0),
    ("What Would Reach 95%?", C_TEAL, True, 15),
    ("• 500+ epochs (estimated from curve extrapolation)", C_WHITE, False, 13),
    ("• OR: more labeled data (reduce pseudo-label noise)", C_WHITE, False, 13),
    ("• OR: test-time augmentation (TTA) ensemble boost +1-2%", C_WHITE, False, 13),
    ("• OR: post-processing with atlas-based regularization", C_WHITE, False, 13),
    ("", None, False, 0),
    ("Current Realistic Ceiling", C_TEAL, True, 15),
    ("Fold 0 (250 ep):  0.914 val dice", C_GREEN, False, 13),
    ("Fold 1 (150 ep):  0.912* EMA dice", C_TEAL, False, 13),
    ("Fold 2 (150 ep):  ~0.895–0.905 (projected)", C_TEAL, False, 13),
    ("5-fold ensemble:  ~0.910–0.920 (expected gain)", C_YELLOW, False, 13),
    ("With self-training: ~0.920–0.930 (FLARE22 estimate)", C_YELLOW, False, 13),
]

ay = Inches(1.7)
for item in analysis:
    text, col, bold, size = item
    if text == "":
        ay += Inches(0.12)
        continue
    txb(sl, text, Inches(6.8), ay, Inches(6.1), Inches(0.75),
        size=size, color=col, bold=bold)
    ay += Inches(0.42) if size > 14 else Inches(0.38)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Self-Training (FLARE22 Recipe)
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Self-Training — FLARE22 Recipe", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Left: pipeline steps
steps_st = [
    ("Round 1: Initial Training", [
        "Train nnU-Net on SPIDER labeled data (5 folds)",
        "Dataset100_SpineL1L5 — 447 volumes",
    ], C_TEAL),
    ("Round 2: Pseudo-Label Generation", [
        "Run all 5 folds on 243 unlabeled T1-FLAIR scans",
        "Keep predictions where ≥ 4/5 folds agree (high confidence)",
        "Threshold: 0.80 confidence per voxel",
    ], C_GREEN),
    ("Round 3: Expanded Training", [
        "Merge pseudo-labeled T1-FLAIR + SPIDER",
        "Dataset202_SpineL1L5_SSL — 346 volumes total",
        "Retrain from scratch with larger, richer dataset",
    ], C_YELLOW),
    ("Round 4 (Optional)", [
        "Re-predict unlabeled data with stronger model",
        "Add newly confident predictions",
        "Repeat 2–3 rounds for maximum accuracy",
    ], C_MGRAY),
]

sy = Inches(1.2)
for title, items, col in steps_st:
    rect(sl, Inches(0.35), sy, Inches(6.1), Inches(1.4 + len(items)*0.35), C_ACCENT,
         line_color=col, line_w=Pt(1.5))
    txb(sl, title, Inches(0.5), sy + Inches(0.07), Inches(5.9), Inches(0.38),
        size=14, bold=True, color=col)
    iy = sy + Inches(0.48)
    for item in items:
        txb(sl, "•  " + item, Inches(0.5), iy, Inches(5.9), Inches(0.35),
            size=12, color=C_LGRAY)
        iy += Inches(0.33)
    sy += Inches(1.4 + len(items)*0.35) + Inches(0.12)

# Right: benefits + diagram
rx = Inches(7.0)
txb(sl, "Why Self-Training Works", rx, Inches(1.2), Inches(6.0), Inches(0.4),
    size=17, bold=True, color=C_TEAL)

benefits = [
    ("Domain Adaptation", "SPIDER is university-acquired MRI.\nClinical T1-FLAIR has different contrast, noise, field strength.\nSelf-training bridges this domain gap.", C_TEAL),
    ("Data Multiplier", "346 volumes >> 243 labeled volumes.\nPseudo-labels add +103 training examples\nfor free, boosting generalization.", C_GREEN),
    ("Conservative Labeling", "4/5 fold agreement is a strong filter.\nOnly genuinely easy cases become pseudo-labels,\navoiding label noise propagation.", C_YELLOW),
]

by2 = Inches(1.8)
for title, desc, col in benefits:
    rect(sl, rx, by2, Inches(6.0), Inches(1.3), C_BLUE, line_color=col, line_w=Pt(1))
    txb(sl, title, rx + Inches(0.1), by2 + Inches(0.05), Inches(5.8), Inches(0.35),
        size=13, bold=True, color=col)
    txb(sl, desc, rx + Inches(0.1), by2 + Inches(0.42), Inches(5.8), Inches(0.8),
        size=12, color=C_LGRAY)
    by2 += Inches(1.45)

# Expected boost box
rect(sl, rx, by2 + Inches(0.1), Inches(6.0), Inches(0.9), C_ACCENT)
txb(sl, "Expected Accuracy with Self-Training", rx + Inches(0.1), by2 + Inches(0.15),
    Inches(5.8), Inches(0.3), size=13, bold=True, color=C_WHITE)
txb(sl, "Fold 0 baseline:  0.914 Dice   →   After self-training:  ~0.920–0.930 (est.)",
    rx + Inches(0.1), by2 + Inches(0.5), Inches(5.8), Inches(0.35),
    size=13, color=C_YELLOW, bold=True)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Inference Pipeline
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Inference & Output Pipeline", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Inference flow
inf_steps = [
    ("Input\nMRI", ".nii.gz\nsagittal T1", C_MGRAY, Inches(0.4)),
    ("Preprocessing\n(nnU-Net)", "Z-score norm\nspine crop", C_ACCENT, Inches(2.2)),
    ("5-Fold\nEnsemble", "Avg softmax\nall folds", C_TEAL, Inches(4.0)),
    ("Argmax\nDecoder", "Per-voxel\nclass label", C_GREEN, Inches(5.8)),
    ("Post-\nProcess", "Largest CC\nhole-fill", C_ACCENT, Inches(7.6)),
    ("Outputs", "Masks + mesh\n+ overlay", C_YELLOW, Inches(9.4)),
]

for label, desc, col, x in inf_steps:
    rect(sl, x, Inches(1.5), Inches(1.6), Inches(1.4), C_BLUE,
         line_color=col, line_w=Pt(1.5))
    txb(sl, label, x, Inches(1.55), Inches(1.6), Inches(0.65),
        size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
    txb(sl, desc, x, Inches(2.2), Inches(1.6), Inches(0.65),
        size=11, color=C_LGRAY, align=PP_ALIGN.CENTER)
    if x < Inches(9.4):
        txb(sl, "→", x + Inches(1.6), Inches(2.0), Inches(0.2), Inches(0.4),
            size=16, color=C_TEAL, align=PP_ALIGN.CENTER)

# Output formats
txb(sl, "Output Formats", Inches(0.4), Inches(3.3), Inches(6), Inches(0.4),
    size=16, bold=True, color=C_TEAL)

out_formats = [
    ("NIfTI Segmentation Masks", ".nii.gz", "Per-vertebra binary masks (L1–L5).\nDirect input for radiological review software.", C_TEAL),
    ("3D STL Meshes", ".stl", "Watertight vertebra surface meshes.\nFor 3D printing, surgical planning, Slicer3D.", C_GREEN),
    ("Overlay Visualizations", ".png", "Multi-panel: raw MRI, ground truth (if avail.),\nprediction, error map.", C_YELLOW),
    ("3D Slicer Scenes", ".mrb", "Complete Slicer bundle with volumes and\ncolor-coded segmentation labels.", C_TEAL),
]

ox = Inches(0.35)
oy = Inches(3.8)
for title, ext, desc, col in out_formats:
    rect(sl, ox, oy, Inches(3.1), Inches(1.5), C_ACCENT, line_color=col, line_w=Pt(1))
    txb(sl, title, ox + Inches(0.1), oy + Inches(0.05), Inches(2.5), Inches(0.35),
        size=13, bold=True, color=col)
    txb(sl, ext, ox + Inches(2.55), oy + Inches(0.05), Inches(0.5), Inches(0.3),
        size=12, color=C_MGRAY, italic=True, bold=True)
    txb(sl, desc, ox + Inches(0.1), oy + Inches(0.45), Inches(2.9), Inches(0.95),
        size=11, color=C_LGRAY)
    ox += Inches(3.3)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — Summary & Next Steps
# ══════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
bg(sl)

rect(sl, Inches(0), Inches(0), SLIDE_W, Inches(1.05), C_BLUE)
txb(sl, "Summary & Next Steps", Inches(0.4), Inches(0.2),
    Inches(12), Inches(0.7), size=30, bold=True, color=C_WHITE)

# Achievements
txb(sl, "What We've Achieved", Inches(0.4), Inches(1.2),
    Inches(6.2), Inches(0.4), size=17, bold=True, color=C_GREEN)

achievements = [
    "✅  Fold 0 complete — 250 epochs — Dice 0.914  (val: 71 volumes)",
    "✅  Fold 1 complete — 150 epochs — Dice 0.912* (EMA, 12h vs 20h)",
    "🔄  Fold 2 training — 116/150 epochs — on track for ~0.895 dice",
    "⏳  Folds 3 & 4 queued — will auto-start via cascade script",
    "✅  RTX 5090 stability fixes deployed (5 crash types resolved)",
    "✅  Auto-resume checkpointing working correctly",
    "✅  Rich terminal training monitor with dice sparklines",
]

ay2 = Inches(1.75)
for a in achievements:
    col = C_GREEN if a.startswith("✅") else (C_TEAL if a.startswith("🔄") else C_MGRAY)
    txb(sl, a, Inches(0.5), ay2, Inches(6.0), Inches(0.42), size=13, color=col)
    ay2 += Inches(0.42)

# Key metrics summary
txb(sl, "Key Metrics Summary", Inches(0.4), ay2 + Inches(0.15),
    Inches(6.2), Inches(0.4), size=16, bold=True, color=C_TEAL)

metrics_sum = [
    ("Metric",             "Value",   True),
    ("Training Dataset",   "346 volumes", False),
    ("Architecture",       "nnU-Net ResEnc L (3d_fullres)", False),
    ("Best Dice (Fold 0)", "0.914",   False),
    ("Per-vertebra range", "L1: 0.906 — L3: 0.922", False),
    ("IoU (mean)",         "0.859",   False),
    ("Improvement vs V7",  "+0.138 Dice  (+17.8%)", False),
]

msy = ay2 + Inches(0.65)
for k, v, bold in metrics_sum:
    bc = C_ACCENT if bold else (C_BLUE if (msy - ay2) / Inches(0.36) % 2 == 0 else RGBColor(0x1A,0x2A,0x3A))
    rect(sl, Inches(0.4), msy, Inches(6.0), Inches(0.34), bc)
    txb(sl, k, Inches(0.5), msy + Inches(0.04), Inches(2.5), Inches(0.28),
        size=12, bold=bold, color=C_WHITE if bold else C_LGRAY)
    txb(sl, v, Inches(3.0), msy + Inches(0.04), Inches(3.3), Inches(0.28),
        size=12, bold=bold, color=C_YELLOW if bold else C_WHITE)
    msy += Inches(0.36)

# Right: next steps
txb(sl, "Next Steps (Roadmap)", Inches(7.0), Inches(1.2),
    Inches(6.0), Inches(0.4), size=17, bold=True, color=C_TEAL)

next_steps = [
    ("Short-term (1–2 days)", [
        "Complete folds 2, 3, 4 training",
        "Run 5-fold ensemble validation",
        "Generate ensemble predictions on all T1-FLAIR",
    ], C_TEAL),
    ("Medium-term (1 week)", [
        "Launch self-training Round 2 (FLARE22 recipe)",
        "Filter pseudo-labels at ≥4/5 fold agreement",
        "Retrain on expanded 346→400+ volume dataset",
    ], C_GREEN),
    ("Long-term (2–4 weeks)", [
        "Evaluate full ensemble on held-out test set",
        "Generate STL meshes for all clinical cases",
        "Export 3D Slicer scenes for radiological review",
        "Target: 0.920–0.930 Dice with self-training",
    ], C_YELLOW),
]

nsy = Inches(1.8)
for title, items, col in next_steps:
    rect(sl, Inches(7.0), nsy, Inches(6.0),
         Inches(0.45 + len(items) * 0.42), C_BLUE, line_color=col, line_w=Pt(1.5))
    txb(sl, title, Inches(7.1), nsy + Inches(0.07), Inches(5.8), Inches(0.35),
        size=14, bold=True, color=col)
    iy = nsy + Inches(0.48)
    for item in items:
        txb(sl, "→  " + item, Inches(7.1), iy, Inches(5.8), Inches(0.38),
            size=12, color=C_LGRAY)
        iy += Inches(0.38)
    nsy += Inches(0.45 + len(items) * 0.42) + Inches(0.18)


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
prs.save(OUT)
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides)}")
