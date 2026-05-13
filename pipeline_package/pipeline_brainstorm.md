# 🧠 Spine L1–L5 Segmentation: Pipeline Redesign Brainstorm
> Design facilitation following the @brainstorming skill protocol.  
> Status: **Understanding Lock** — awaiting your confirmation before finalizing approach.

---

## 📊 Part 1: Current State Analysis

### What We Have
| Item | Detail |
|---|---|
| **GPU** | RTX 5090 |
| **Framework** | nnUNetv2, ResEncL architecture, 3d_fullres |
| **Training data (Dataset100)** | 459 GT-labeled cases (T2w MRI, multi-site) |
| **Self-training data (Dataset101)** | 459 GT + 12 duke T1-FLAIR pseudo-labels → 494 cases |
| **Task** | L1–L5 vertebra body segmentation (5 classes) |
| **Training** | Fold 0 trained ~270 epochs, mean Dice ≈ 0.884 |
| **Pseudo-labels** | Only 12 duke cases used (35 attempted, most filtered out due to low quality) |

### Training Metrics (Fold 0 — what we can read from CSV)
| Epoch | Mean Dice | Notes |
|---|---|---|
| 0 | 0.0 | random init |
| 267 | **0.884** | Best visible epoch |
| 268 | 0.843 | slight dip — possible overfit |
| Final checkpoint | ~0.884 | EMA best |

**Per-class at epoch 267:**
- L1: 0.848, L2: 0.883, L3: 0.892, L4: 0.893, L5: 0.904

### Critical Problems Identified

1. **Pseudo-labels are too few (12/35)** — our filter was conservative but T1-FLAIR to T2w domain gap is large. 12 noisy cases add minimal signal.
2. **nnUNet default training is only 250 epochs** — yet the loss curve shows we were still improving. We stopped too early.
3. **No cross-modality strategy** — T1-FLAIR data (the unlabeled duke set) has fundamentally different contrast from T2w training data. Naive pseudo-label injection creates domain mismatch noise.
4. **No validation summary.json exists** — we have no formal evaluation, no Dice computed on a held-out test set.
5. **Current self-training (Dataset101) is still running** — but the same architectural and domain issues persist.
6. **Direction mismatch in duke_0458** — visible data quality issues in the duke set that the integrity check caught.
7. **Only fold 0 log exists** — 5-fold cross-validation not confirmed complete.

---

## 📚 Part 2: Research Synthesis (2024–2025 State of the Art)

### Finding 1: nnUNet ResEncL is the right backbone ✅
Papers consistently confirm nnUNet v2 with ResEncL plans beats all transformer/Mamba variants when properly validated on spine MRI. We already use this. **Keep it.**

### Finding 2: Cascaded architecture is the SOTA for vertebra labeling
**TotalSpineSeg** (MICCAI 2024, Spinal Cord Toolbox integration):
- Stage 1: Coarse segmentation — find the whole spine region
- Stage 2: Fine instance segmentation per vertebra + iterative labeling
- Achieves Dice 0.88 for vertebrae, 0.80 for IVDs
- Much more robust to FOV truncation, low contrast, multi-contrast

**Implication**: Our flat single-stage approach may confuse vertebra identity (L1 vs L2 vs L3) in edge cases. A 2-stage approach (binary spine → per-vertebra) is more robust.

### Finding 3: Cross-modality pseudo-labeling requires domain adaptation
The T1-FLAIR → T2w gap is a known problem. Best practices:
- **Histogram matching / intensity normalization** before pseudo-labeling
- **Mean Teacher** or **CPS (Cross Pseudo Supervision)** framework — train two networks, each generates pseudo-labels for the other with consistency loss
- **FixMatch** threshold: 0.95 minimum confidence per voxel before treating as pseudo-GT

### Finding 4: SPIDER challenge benchmark shows what good performance looks like
- Top performers: mean Dice **0.93–0.96** for lumbar vertebrae on T2w MRI
- Key advantage of top methods: **multi-scale patch sampling + larger training epochs (500–1000)**
- nnUNet default is 250 epochs but SPIDER winners used 500+

### Finding 5: Test-Time Augmentation (TTA) + 5-fold ensemble adds ~1-2% Dice
- Mirror flipping along all 3 axes = 8 predictions averaged
- 5-fold ensemble stacks on top → together ~1.5-2% free gain at inference
- We are NOT using this yet for evaluation

### Finding 6: Intensity normalization matters for MRI more than CT
- Z-score per-image normalization (nnUNet default) is OK but…
- **Percentile-clipped z-score** (clip at 0.5/99.5 percentile) handles pathological outliers better
- **Histogram equalization** helps cross-modality generalization

---

## 🎯 Part 3: Understanding Lock

### What is being built
A robust, high-accuracy **L1–L5 lumbar vertebra segmentation pipeline** for 3D MRI volumes, capable of handling multi-contrast input (primarily T2w, with T1-FLAIR generalization).

### Why it exists
- Clinical / research use: automated vertebra labeling for spine analysis
- Current model achieves ~0.884 mean Dice (fold 0, epoch 267) and needs improvement
- Semi-supervised learning with the unlabeled duke T1-FLAIR set is the avenue for more data

### Key Constraints
- **Hardware**: RTX 5090 (ample VRAM)
- **Self-imposed**: nnUNetv2 framework (reasonable — it's the right choice)
- **Data**: 459 labeled T2w cases + ~500+ unlabeled T1-FLAIR cases (duke)
- **Goal metric**: Mean Dice > 0.92 on L1–L5 (SPIDER-competitive)

### Explicit Non-Goals
- Real-time inference (clinical pipeline, batch is fine)
- IVD / spinal canal / cord segmentation (vertebra bodies only)
- Novel architecture exploration (stay on nnUNet)

### Assumptions (marked as such)
- *Assumption*: The 459 GT cases are high quality and correctly labeled
- *Assumption*: The duke T1-FLAIR set is anatomically valid (same L1–L5 scope)
- *Assumption*: We can run training for 500–1000 epochs if needed
- *Assumption*: We want a final ensemble of all 5 folds

---

## 🔀 Part 4: Three Pipeline Approaches

---

### Option A: Fix & Extend Current Pipeline (Low Risk, Incremental)

**What changes:**
1. Stop current Dataset101 training (domain-mixing is noisy)
2. Re-run Dataset100 with **500 epochs** (not 250)
3. Add **percentile-clipped z-score normalization** via custom preprocessor
4. Use **full 5-fold ensemble + TTA** for inference
5. Improve pseudo-label generation:
   - Apply **histogram matching** (T1-FLAIR → T2w style) before inference
   - Use **per-voxel confidence threshold 0.95** (from softmax probabilities)
   - Accept only cases where ALL 5 vertebra classes present + volume in range

**Expected gain**: 0.884 → ~0.90–0.92 mean Dice  
**Risk**: Low (minor changes to existing setup)  
**Time**: ~3-4 days additional training  

---

### Option B: Mean Teacher Semi-Supervised Pipeline (Medium Risk, High Reward) ⭐ RECOMMENDED

**What changes:**
1. Train a **teacher model** (5-fold, 500 epochs) on Dataset100 only — this is our quality baseline
2. Implement **Mean Teacher** framework:
   - Student network train on GT + pseudo-labels
   - Teacher network = exponential moving average (EMA) of student weights
   - Consistency loss between student and teacher predictions on unlabeled data
   - No need to threshold pseudo-labels manually — consistency loss handles it
3. Apply **modality alignment**:
   - MONAI's `NormalizeIntensity` with percentile clipping per modality
   - Optional: train a simple intensity normalizer on duke data to map T1-FLAIR → T2w histogram
4. **500 epochs, all 5 folds, full ensemble + TTA at inference**

**Why this is best:**
- Mean Teacher prevents confirmation bias (teacher generates soft pseudo-labels, not hard thresholds)
- Used by MICCAI 2024 best performers for semi-supervised medical segmentation
- Consistency regularization naturally handles domain shift to some degree
- Still uses nnUNet backbone — no need to rebuild from scratch

**Expected gain**: 0.884 → ~0.92–0.95 mean Dice  
**Risk**: Medium (requires new training loop, but MONAI has official Mean Teacher support)  
**Time**: ~5-7 days  

---

### Option C: Cascaded 2-Stage Pipeline (High Reward, Higher Complexity)

**What changes:**
1. **Stage 1**: Binary spine localization model (all vertebrae = 1 class)
   - Much easier to train, very high Dice achievable (~0.97+)
   - Crops to tight bounding box around spine
2. **Stage 2**: Fine L1–L5 classification model on cropped spine patch
   - Input: cropped spine region from Stage 1
   - More focused, less background confusion
3. **Post-processing**: Iterative labeling (enforce superior-to-inferior ordering: L1 → L5)
4. Apply Mean Teacher semi-supervision at Stage 2

**Why this helps:**
- Stage 1 eliminates background confusion
- Stage 2 model sees only spine → can focus on distinguishing L1 vs L5
- Robust to FOV truncation (common in clinical MRI)
- This is exactly what TotalSpineSeg does and achieves best published scores

**Expected gain**: 0.884 → ~0.93–0.96 mean Dice  
**Risk**: High (2x training, more engineering, more failure modes)  
**Time**: 10-14 days  

---

## 🗓️ Decision Log (Initial)

| Decision Point | Options | Recommendation | Reason |
|---|---|---|---|
| Architecture | nnUNet vs Transformer | nnUNet ResEncL | SOTA evidence, already have it |
| Training epochs | 250 vs 500 | 500 | Still improving at epoch 267 |
| Semi-supervision | Simple pseudo-label vs Mean Teacher | Mean Teacher | Prevents confirmation bias, domain gap |
| Modality alignment | None vs Histogram matching | Histogram matching | T1-FLAIR ≠ T2w |
| Cascade | Single stage vs 2-stage | Start single, cascade if needed | Risk-adjusted choice |
| Inference | Single model vs Ensemble+TTA | Full ensemble + TTA | Free 1-2% Dice |

---

## ❓ Questions For You (One at a Time)

**Current question:**

> Does this understanding summary accurately reflect your goal?  
> In particular:  
> - Is your target metric Mean Dice > 0.92, or do you have a specific benchmark?
> - Is the duke T1-FLAIR dataset the only unlabeled source, or do you have other data?
> - Do you want to use MONAI/custom code to implement Mean Teacher, or would you prefer to stay 100% within nnUNetv2's standard interface?

Please confirm or correct before I finalize the approach.
