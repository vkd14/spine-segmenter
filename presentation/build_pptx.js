// Build the spine-segmentation term-project deck.
// Run:  node build_pptx.js
// Output: spine_seg_presentation.pptx

const path = require("path");
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3" x 7.5" — gives us breathing room
pres.author = "Varun Dasoju";
pres.title = "Lumbar L1-L5 Vertebral Body Segmentation";
pres.subject = "Term Project — Image Processing";
pres.company = "Image Processing";

// ---- Slide dimensions ------------------------------------------------------
const W = 13.3;
const H = 7.5;

// ---- Color palette (Ocean Gradient + L1-L5 vertebra brand colors) ---------
const C = {
  primary:   "065A82", // deep blue
  secondary: "1C7293", // teal
  accent:    "21295C", // midnight
  bgDark:    "0E1B2C", // deep navy
  bgLight:   "F7FAFC", // off-white
  ink:       "1A202C", // body text
  inkMute:   "4A5568", // captions
  rule:      "CBD5E0", // hairline divider
  l1: "DC2626",  l2: "16A34A",  l3: "2563EB",  l4: "EA580C",  l5: "9333EA",
};

// Path to figures
const FIG = (name) => path.join(__dirname, "figures", name);

// =============================================================================
// Helpers
// =============================================================================

function addTitleBar(slide, title, sectionLabel = null) {
  // Vertical accent bar on the left of the title
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.45, w: 0.08, h: 0.55,
    fill: { color: C.primary }, line: { type: "none" }
  });
  // Title text
  slide.addText(title, {
    x: 0.7, y: 0.4, w: W - 1.4, h: 0.65,
    fontFace: "Calibri", fontSize: 30, bold: true,
    color: C.ink, valign: "middle", margin: 0,
  });
  if (sectionLabel) {
    slide.addText(sectionLabel.toUpperCase(), {
      x: 0.7, y: 0.18, w: W - 1.4, h: 0.25,
      fontFace: "Calibri", fontSize: 10.5, bold: true,
      color: C.secondary, charSpacing: 4, valign: "middle", margin: 0,
    });
  }
  // Hairline divider under title
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 1.15, w: W - 1.0, h: 0,
    line: { color: C.rule, width: 0.75 }
  });
}

function addFooter(slide, slideNum, total) {
  slide.addText(`${slideNum} / ${total}`, {
    x: W - 1.2, y: H - 0.45, w: 0.7, h: 0.3,
    fontFace: "Calibri", fontSize: 9, color: C.inkMute,
    align: "right", valign: "middle", margin: 0,
  });
  slide.addText("L1–L5 Vertebral Body Segmentation", {
    x: 0.5, y: H - 0.45, w: 8.0, h: 0.3,
    fontFace: "Calibri", fontSize: 9, color: C.inkMute,
    align: "left", valign: "middle", margin: 0, italic: true,
  });
}

// Make a new content slide (light background, title bar, footer)
function newContentSlide(title, sectionLabel, slideNum, total) {
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };
  addTitleBar(slide, title, sectionLabel);
  addFooter(slide, slideNum, total);
  return slide;
}

// Bullet list helper — uses pptxgenjs bullets (never unicode)
function bulletList(slide, items, opts) {
  const arr = items.map((t, i) => ({
    text: t,
    options: {
      bullet: { code: "25A0" },     // small filled square
      breakLine: i < items.length - 1,
      paraSpaceAfter: 6,
    },
  }));
  slide.addText(arr, {
    fontFace: "Calibri", fontSize: opts.fontSize || 16,
    color: C.ink, ...opts,
  });
}

// Colored swatch + label (used in palette / legend rows)
function swatchRow(slide, x, y, w, h, color, label) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y + 0.05, w: 0.22, h: 0.22,
    fill: { color: color }, line: { type: "none" },
  });
  slide.addText(label, {
    x: x + 0.32, y: y, w: w - 0.32, h: h,
    fontFace: "Calibri", fontSize: 13, color: C.ink,
    valign: "middle", margin: 0,
  });
}

// Tiny pill (used in title slide for the chips)
function chip(slide, x, y, label, color) {
  const w = 0.14 * label.length + 0.4;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: w, h: 0.34,
    fill: { color: color }, line: { type: "none" }, rectRadius: 0.17,
  });
  slide.addText(label, {
    x: x, y: y, w: w, h: 0.34,
    fontFace: "Calibri", fontSize: 11, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
  return w;
}

// Box used in flow diagrams
function flowBox(slide, x, y, w, h, lines, kind = "primary") {
  const palette = {
    primary: { fill: "E6EEF5", border: C.primary, text: C.primary },
    accent:  { fill: "FFF1E6", border: C.l4,      text: C.l4 },
    output:  { fill: "E6F4EC", border: C.l2,      text: "166534" },
  }[kind];
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: w, h: h,
    fill: { color: palette.fill },
    line: { color: palette.border, width: 1.25 },
    rectRadius: 0.08,
  });
  slide.addText(lines, {
    x: x, y: y, w: w, h: h,
    fontFace: "Calibri", fontSize: 11.5, bold: true,
    color: palette.text, align: "center", valign: "middle",
    margin: 4,
  });
}

function arrow(slide, x1, y1, x2, y2) {
  slide.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: C.inkMute, width: 1.25, endArrowType: "triangle" },
  });
}

// ============================================================================
// SLIDE 1 — Title
// ============================================================================

const TOTAL = 21;
let slideNum = 0;

(function titleSlide() {
  const slide = pres.addSlide();
  slide.background = { color: C.bgDark };

  // Top accent stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.18,
    fill: { color: C.primary }, line: { type: "none" }
  });

  // L1-L5 vertebra dots, top-right (visual motif)
  const dotY = 0.45;
  [C.l1, C.l2, C.l3, C.l4, C.l5].forEach((c, i) => {
    slide.addShape(pres.shapes.OVAL, {
      x: W - 2.6 + i * 0.45, y: dotY, w: 0.32, h: 0.32,
      fill: { color: c }, line: { type: "none" },
    });
  });

  slide.addText("TERM PROJECT  ·  IMAGE PROCESSING", {
    x: 0.7, y: 1.4, w: 12, h: 0.4,
    fontFace: "Calibri", fontSize: 13, bold: true,
    color: "8FB8DA", charSpacing: 6, margin: 0,
  });

  slide.addText("Automated Lumbar Vertebral", {
    x: 0.7, y: 1.95, w: 12, h: 1.0,
    fontFace: "Cambria", fontSize: 46, bold: true,
    color: "FFFFFF", margin: 0,
  });
  slide.addText("Body Segmentation", {
    x: 0.7, y: 2.85, w: 12, h: 1.0,
    fontFace: "Cambria", fontSize: 46, bold: true,
    color: "FFFFFF", margin: 0,
  });

  // Subtitle bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 4.05, w: 0.06, h: 0.7,
    fill: { color: C.l3 }, line: { type: "none" }
  });
  slide.addText([
    { text: "An End-to-End ML and Image Processing Pipeline", options: { breakLine: true, fontSize: 18, bold: true, color: "E2ECF7" } },
    { text: "From raw T1 FLAIR MRI to interactive 3D predictions", options: { fontSize: 14, italic: true, color: "8FB8DA" } },
  ], {
    x: 0.95, y: 4.05, w: 11, h: 0.85, fontFace: "Calibri", margin: 0,
  });

  // Tag chips
  let cx = 0.7;
  cx += chip(slide, cx, 5.25, "nnU-Net v2", C.primary) + 0.18;
  cx += chip(slide, cx, 5.25, "ResEncUNet Large", C.secondary) + 0.18;
  cx += chip(slide, cx, 5.25, "5-fold Ensemble", C.l3) + 0.18;
  cx += chip(slide, cx, 5.25, "STL · NIfTI · Slicer", C.l4) + 0.18;
  cx += chip(slide, cx, 5.25, "Native macOS App", C.l5) + 0.18;

  // Author block bottom-left
  slide.addText("VARUN DASOJU", {
    x: 0.7, y: H - 1.0, w: 6, h: 0.4,
    fontFace: "Calibri", fontSize: 14, bold: true,
    color: "FFFFFF", charSpacing: 4, margin: 0,
  });
  slide.addText(`${new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}`, {
    x: 0.7, y: H - 0.65, w: 6, h: 0.3,
    fontFace: "Calibri", fontSize: 11, italic: true,
    color: "8FB8DA", margin: 0,
  });

  // Bottom accent stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.18, w: W, h: 0.18,
    fill: { color: C.primary }, line: { type: "none" }
  });
})();

// ============================================================================
// SLIDE 2 — Agenda
// ============================================================================

(function agendaSlide() {
  slideNum = 2;
  const slide = newContentSlide("Agenda", null, slideNum, TOTAL);

  const items = [
    { num: "01", title: "Motivation",            sub: "Clinical relevance and bottleneck" },
    { num: "02", title: "Pipeline Overview",     sub: "End-to-end flow at a glance" },
    { num: "03", title: "Data",                  sub: "459 sagittal T1 FLAIR cases, six classes" },
    { num: "04", title: "Model Architecture",    sub: "nnU-Net + ResEncUNet Large" },
    { num: "05", title: "Training",              sub: "5-fold cross-validation, 1000 epochs" },
    { num: "06", title: "Inference & Post-Proc", sub: "Three deterministic anatomical rules" },
    { num: "07", title: "Mesh Extraction",       sub: "Marching cubes + Taubin smoothing" },
    { num: "08", title: "Application",           sub: "Streamlit + native macOS window" },
    { num: "09", title: "Results",               sub: "MPR overlay, 3D meshes, voxel counts" },
    { num: "10", title: "Limitations & Wrap-up", sub: "Edge cases and conclusion" },
  ];

  const colors = [C.l1, C.l2, C.l3, C.l4, C.l5, C.primary, C.secondary, C.l4, C.l3, C.l5];

  // Two columns of cards
  items.forEach((it, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.7 + col * 6.0;
    const y = 1.4 + row * 1.05;
    // Number circle
    slide.addShape(pres.shapes.OVAL, {
      x: x, y: y + 0.05, w: 0.7, h: 0.7,
      fill: { color: colors[i] }, line: { type: "none" },
    });
    slide.addText(it.num, {
      x: x, y: y + 0.05, w: 0.7, h: 0.7,
      fontFace: "Calibri", fontSize: 16, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    });
    // Text
    slide.addText([
      { text: it.title, options: { breakLine: true, fontSize: 15, bold: true, color: C.ink } },
      { text: it.sub, options: { fontSize: 11.5, color: C.inkMute, italic: true } },
    ], {
      x: x + 0.85, y: y, w: 5.0, h: 0.85, fontFace: "Calibri", margin: 0, valign: "middle",
    });
  });
})();

// ============================================================================
// SLIDE 3 — Motivation
// ============================================================================

(function motivationSlide() {
  slideNum = 3;
  const slide = newContentSlide("Why Segment Lumbar Vertebrae?", "01 · Motivation", slideNum, TOTAL);

  // Left column — clinical drivers as icon-rows
  const drivers = [
    { color: C.l1, h: "Spinal stenosis grading",  s: "Cross-sectional canal area per level" },
    { color: C.l2, h: "Vertebral height & bone quality", s: "Quantitative morphometry" },
    { color: C.l3, h: "Pre-surgical planning",    s: "Screw trajectories, anatomical landmarks" },
    { color: C.l4, h: "Bone densitometry from MRI", s: "Per-vertebra signal-intensity analysis" },
  ];
  drivers.forEach((d, i) => {
    const y = 1.55 + i * 1.0;
    // Color tag
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: y, w: 0.08, h: 0.85,
      fill: { color: d.color }, line: { type: "none" },
    });
    slide.addText([
      { text: d.h, options: { breakLine: true, fontSize: 16, bold: true, color: C.ink } },
      { text: d.s, options: { fontSize: 12, color: C.inkMute, italic: true } },
    ], {
      x: 0.95, y: y, w: 5.5, h: 0.85, fontFace: "Calibri", margin: 0, valign: "middle",
    });
  });

  // Right column — bottleneck callout
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.2, y: 1.55, w: 5.6, h: 2.0,
    fill: { color: "FFF7ED" }, line: { color: C.l4, width: 1.5 }, rectRadius: 0.12,
  });
  slide.addText("THE BOTTLENECK", {
    x: 7.4, y: 1.7, w: 5.2, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.l4,
    charSpacing: 4, margin: 0,
  });
  slide.addText("~30 min", {
    x: 7.4, y: 2.05, w: 5.2, h: 0.7,
    fontFace: "Cambria", fontSize: 44, bold: true, color: C.l4,
    margin: 0,
  });
  slide.addText("per case for manual delineation, with high inter-observer variability.", {
    x: 7.4, y: 2.85, w: 5.2, h: 0.55,
    fontFace: "Calibri", fontSize: 13, color: C.ink, italic: true,
    margin: 0, valign: "top",
  });

  // Goal box
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.2, y: 3.85, w: 5.6, h: 2.65,
    fill: { color: "EBF4FA" }, line: { color: C.primary, width: 1.5 }, rectRadius: 0.12,
  });
  slide.addText("OUR GOAL", {
    x: 7.4, y: 4.0, w: 5.2, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary,
    charSpacing: 4, margin: 0,
  });
  slide.addText([
    { text: "Fully-automated, anatomically-valid", options: { breakLine: true, fontSize: 16, bold: true, color: C.ink } },
    { text: "L1–L5 segmentation pipeline driven", options: { breakLine: true, fontSize: 16, bold: true, color: C.ink } },
    { text: "from a single sagittal T1 FLAIR scan,", options: { breakLine: true, fontSize: 16, bold: true, color: C.ink } },
    { text: "with an interactive desktop app for", options: { breakLine: true, fontSize: 16, bold: true, color: C.ink } },
    { text: "inspection and export.", options: { fontSize: 16, bold: true, color: C.ink } },
  ], {
    x: 7.4, y: 4.35, w: 5.2, h: 2.05, fontFace: "Calibri", margin: 0, valign: "top",
  });
})();

// ============================================================================
// SLIDE 4 — Pipeline Overview (flow diagram)
// ============================================================================

(function pipelineSlide() {
  slideNum = 4;
  const slide = newContentSlide("End-to-End Pipeline", "02 · Pipeline Overview", slideNum, TOTAL);

  // Top row: Raw -> Stage 1 -> Stage 2
  const topY = 1.55;
  flowBox(slide, 0.7,   topY, 2.6, 0.9, "Raw T1 FLAIR\nNIfTI volume", "primary");
  arrow(slide, 3.3,   topY + 0.45, 3.7, topY + 0.45);
  flowBox(slide, 3.7,   topY, 2.6, 0.9, "Stage 1\nInput Preparation", "primary");
  arrow(slide, 6.3,   topY + 0.45, 6.7, topY + 0.45);
  flowBox(slide, 6.7,   topY, 2.6, 0.9, "Stage 2\nnnU-Net Inference\n(1- or 5-fold)", "primary");
  arrow(slide, 9.3,   topY + 0.45, 9.7, topY + 0.45);
  flowBox(slide, 9.7,   topY, 2.9, 0.9, "Stage 3\nAnatomical\nPost-Processing", "accent");

  // Vertical drop from post-processing
  arrow(slide, 11.15, topY + 0.9, 11.15, topY + 1.4);

  // Middle row: 3 post-processing steps
  const midY = 3.05;
  flowBox(slide, 9.3,   midY, 3.3, 0.9, "Step 1\nSagittal-strip corridor (±20% LR)", "accent");
  arrow(slide, 9.3, midY + 0.45, 8.9, midY + 0.45);  // arrow leftward into Step 2
  flowBox(slide, 5.8,   midY, 3.1, 0.9, "Step 2\nLargest CC (≥ 50 voxels)", "accent");
  arrow(slide, 5.8, midY + 0.45, 5.4, midY + 0.45);
  flowBox(slide, 2.0,   midY, 3.4, 0.9, "Step 3\nVertical SI ordering enforcement", "accent");

  // Drop down to outputs
  arrow(slide, 3.7, midY + 0.9, 3.7, midY + 1.4);

  // Bottom row: outputs
  const botY = 4.55;
  flowBox(slide, 0.7,   botY, 3.0, 0.9, "Cleaned NIfTI\n(uint8, RAS, original affine)", "output");
  arrow(slide, 3.7, botY + 0.45, 4.1, botY + 0.45);
  flowBox(slide, 4.1,   botY, 3.0, 0.9, "5 STL meshes\n(LPS world coordinates)", "output");
  arrow(slide, 7.1, botY + 0.45, 7.5, botY + 0.45);
  flowBox(slide, 7.5,   botY, 2.9, 0.9, "Slicer color table\n+ load-instructions README", "output");
  arrow(slide, 10.4, botY + 0.45, 10.8, botY + 0.45);
  flowBox(slide, 10.8,  botY, 1.9, 0.9, "App: 3-pane MPR\n+ Plotly 3D mesh", "output");

  // Caption
  slide.addText([
    { text: "Built on ", options: { fontSize: 13, color: C.ink } },
    { text: "nnU-Net v2", options: { fontSize: 13, bold: true, color: C.primary } },
    { text: " with a Residual-Encoder U-Net (ResEncUNet Large) and ", options: { fontSize: 13, color: C.ink } },
    { text: "three deterministic anatomical rules", options: { fontSize: 13, bold: true, color: C.l4 } },
    { text: " applied directly in voxel space.", options: { fontSize: 13, color: C.ink } },
  ], {
    x: 0.7, y: 6.05, w: 12, h: 0.6, fontFace: "Calibri", margin: 0, italic: true, valign: "middle",
  });
})();

// ============================================================================
// SLIDE 5 — Dataset
// ============================================================================

(function datasetSlide() {
  slideNum = 5;
  const slide = newContentSlide("Dataset & Image Characteristics", "03 · Data", slideNum, TOTAL);

  // Big stat cards across the top
  const statCards = [
    { v: "459",  l: "Annotated training\nT1 FLAIR volumes", color: C.primary },
    { v: "242",  l: "Unseen inference\ncohort", color: C.secondary },
    { v: "5+1",  l: "Classes\n(L1–L5 + bg)", color: C.l3 },
    { v: "5",    l: "Cross-validation\nfolds", color: C.l4 },
  ];
  statCards.forEach((s, i) => {
    const x = 0.7 + i * 3.05;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.45, w: 2.85, h: 0.06,
      fill: { color: s.color }, line: { type: "none" },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.51, w: 2.85, h: 1.55,
      fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 },
    });
    slide.addText(s.v, {
      x: x, y: 1.55, w: 2.85, h: 0.85,
      fontFace: "Cambria", fontSize: 38, bold: true, color: s.color,
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.l, {
      x: x, y: 2.42, w: 2.85, h: 0.6,
      fontFace: "Calibri", fontSize: 11, color: C.inkMute,
      align: "center", valign: "top", margin: 0,
    });
  });

  // Image properties table (left)
  slide.addText("IMAGE PROPERTIES", {
    x: 0.7, y: 3.35, w: 6, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  slide.addTable(
    [
      [{ text: "Sequence", options: { color: C.inkMute } }, { text: "Sagittal T1 FLAIR", options: { color: C.ink, bold: true } }],
      [{ text: "Median size", options: { color: C.inkMute } }, { text: "130 × 448 × 422 voxels", options: { color: C.ink, bold: true } }],
      [{ text: "Target spacing", options: { color: C.inkMute } }, { text: "0.66 × 0.63 × 0.66 mm", options: { color: C.ink, bold: true } }],
      [{ text: "Normalisation", options: { color: C.inkMute } }, { text: "z-score (zero-mean)", options: { color: C.ink, bold: true } }],
      [{ text: "File format", options: { color: C.inkMute } }, { text: "NIfTI .nii.gz", options: { color: C.ink, bold: true } }],
    ],
    {
      x: 0.7, y: 3.7, w: 6.0,
      colW: [2.0, 4.0],
      fontFace: "Calibri", fontSize: 13,
      border: { type: "none" },
      rowH: 0.42,
    }
  );

  // Label legend (right)
  slide.addText("LABEL COLOR LEGEND", {
    x: 7.2, y: 3.35, w: 5.5, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  const legend = [
    [C.l1, "Label 1 — L1 vertebral body"],
    [C.l2, "Label 2 — L2 vertebral body"],
    [C.l3, "Label 3 — L3 vertebral body"],
    [C.l4, "Label 4 — L4 vertebral body"],
    [C.l5, "Label 5 — L5 vertebral body"],
  ];
  legend.forEach(([c, l], i) => {
    swatchRow(slide, 7.2, 3.78 + i * 0.42, 5.5, 0.4, c, l);
  });
})();

// ============================================================================
// SLIDE 6 — Raw Input Image
// ============================================================================

(function rawInputSlide() {
  slideNum = 6;
  const slide = newContentSlide("Raw Input: Sagittal T1 FLAIR", "03 · Data", slideNum, TOTAL);

  slide.addImage({
    path: FIG("mri_only.png"),
    x: 0.5, y: 1.4, w: 12.3, h: 4.6,
    sizing: { type: "contain", w: 12.3, h: 4.6 },
  });
  slide.addText(
    "Mid-sagittal · mid-coronal · mid-axial slices of an unseen test volume.  Bright cancellous bone and dark cortical rim are characteristic of T1 FLAIR.",
    {
      x: 0.7, y: 6.2, w: 12, h: 0.6,
      fontFace: "Calibri", fontSize: 12, italic: true, color: C.inkMute,
      align: "center", valign: "middle", margin: 0,
    }
  );
})();

// ============================================================================
// SLIDE 7 — Model Architecture
// ============================================================================

(function archSlide() {
  slideNum = 7;
  const slide = newContentSlide("nnU-Net + Residual-Encoder U-Net (Large)", "04 · Model Architecture", slideNum, TOTAL);

  // Left column — rationale
  slide.addText("WHY nnU-Net?", {
    x: 0.7, y: 1.4, w: 6, h: 0.3, fontFace: "Calibri",
    fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  bulletList(slide, [
    "Self-configures patch size, spacing, batch",
    "State of the art on medical 3D segmentation",
    "Avoids manual hyper-parameter tuning",
  ], { x: 0.7, y: 1.75, w: 6, h: 1.4, fontSize: 14 });

  slide.addText("WHY ResEncUNet (vs vanilla U-Net)?", {
    x: 0.7, y: 3.25, w: 6, h: 0.3, fontFace: "Calibri",
    fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  bulletList(slide, [
    "Residual blocks in the encoder path",
    "Better gradient flow in deeper stages",
    "Up to 6 residual blocks per stage",
    "Captures fine bone edges + large-scale spinal context",
  ], { x: 0.7, y: 3.6, w: 6, h: 2.5, fontSize: 14 });

  // Right column — hyperparameter card
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.0, y: 1.4, w: 5.8, h: 5.1,
    fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.12,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 7.0, y: 1.4, w: 5.8, h: 0.5,
    fill: { color: C.primary }, line: { type: "none" },
  });
  slide.addText("HYPER-PARAMETERS", {
    x: 7.0, y: 1.4, w: 5.8, h: 0.5,
    fontFace: "Calibri", fontSize: 12, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", charSpacing: 4, margin: 0,
  });

  const rows = [
    ["Network class",    "ResidualEncoderUNet"],
    ["Conv type",        "3D (3 × 3 × 3)"],
    ["Stages",           "7"],
    ["Features / stage", "32, 64, 128, 256, 320, 320, 320"],
    ["Residual blocks",  "1, 3, 4, 6, 6, 6, 6"],
    ["Decoder convs",    "1 per stage"],
    ["Normalisation",    "InstanceNorm 3D"],
    ["Activation",       "Leaky ReLU"],
    ["Patch size",       "80 × 320 × 256"],
    ["Batch size",       "2"],
  ];
  slide.addTable(
    rows.map(([k, v]) => [
      { text: k, options: { color: C.inkMute, bold: false } },
      { text: v, options: { color: C.ink, bold: true, fontFace: "Consolas", fontSize: 11 } },
    ]),
    {
      x: 7.2, y: 2.05, w: 5.4,
      colW: [2.1, 3.3],
      fontFace: "Calibri", fontSize: 12,
      rowH: 0.42,
      border: { type: "solid", pt: 0.5, color: "EDF2F7" },
    }
  );
})();

// ============================================================================
// SLIDE 8 — Training Configuration
// ============================================================================

(function trainSlide() {
  slideNum = 8;
  const slide = newContentSlide("Training Configuration", "05 · Training", slideNum, TOTAL);

  // Two equal columns
  // Left card — Optimisation
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 1.4, w: 5.8, h: 5.1,
    fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.12,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 1.4, w: 0.08, h: 5.1,
    fill: { color: C.primary }, line: { type: "none" },
  });
  slide.addText("OPTIMISATION", {
    x: 0.95, y: 1.55, w: 5.4, h: 0.4,
    fontFace: "Calibri", fontSize: 12, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  bulletList(slide, [
    "SGD with Nesterov momentum (μ = 0.99)",
    "Polynomial learning-rate decay, η₀ = 10⁻²",
    "Compound Dice + Cross-Entropy loss",
    "1000 epochs, best-Dice checkpoint",
    "nnU-Net default augmentations:\nrotation, scale, mirror, gamma, elastic",
  ], { x: 0.95, y: 2.05, w: 5.4, h: 4.3, fontSize: 14 });

  // Right card — Cross-validation + inference modes
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 1.4, w: 6.0, h: 2.4,
    fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.12,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.8, y: 1.4, w: 0.08, h: 2.4,
    fill: { color: C.l3 }, line: { type: "none" },
  });
  slide.addText("5-FOLD CROSS-VALIDATION", {
    x: 7.05, y: 1.55, w: 5.6, h: 0.4,
    fontFace: "Calibri", fontSize: 12, bold: true, color: C.l3, charSpacing: 4, margin: 0,
  });
  bulletList(slide, [
    "Five independent models, one per fold",
    "Per-fold checkpoint: checkpoint_best.pth",
    "Inference: average softmax then argmax",
    "Reduces variance, smoother boundaries",
  ], { x: 7.05, y: 2.05, w: 5.6, h: 1.7, fontSize: 13 });

  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 4.0, w: 6.0, h: 2.5,
    fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.12,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.8, y: 4.0, w: 0.08, h: 2.5,
    fill: { color: C.l4 }, line: { type: "none" },
  });
  slide.addText("INFERENCE MODES IN THIS APP", {
    x: 7.05, y: 4.15, w: 5.6, h: 0.4,
    fontFace: "Calibri", fontSize: 12, bold: true, color: C.l4, charSpacing: 4, margin: 0,
  });
  slide.addText([
    { text: "1-fold ", options: { bold: true, color: C.ink, fontSize: 14 } },
    { text: "(fast preview, default on Mac)", options: { color: C.inkMute, fontSize: 13, italic: true, breakLine: true } },
    { text: "≈ 12 minutes per case on M1 Max with MPS.", options: { color: C.inkMute, fontSize: 11.5, italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "5-fold ensemble ", options: { bold: true, color: C.ink, fontSize: 14 } },
    { text: "(best quality)", options: { color: C.inkMute, fontSize: 13, italic: true, breakLine: true } },
    { text: "Trade-off: ~50–60 min per case on the same hardware.", options: { color: C.inkMute, fontSize: 11.5, italic: true } },
  ], {
    x: 7.05, y: 4.55, w: 5.6, h: 1.85, fontFace: "Calibri", margin: 0, valign: "top",
  });
})();

// ============================================================================
// SLIDE 9 — Inference command
// ============================================================================

(function inferenceSlide() {
  slideNum = 9;
  const slide = newContentSlide("Stage 2: Sliding-Window Ensemble Inference", "06 · Inference", slideNum, TOTAL);

  // Code/command block
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 1.4, w: 12.0, h: 2.4,
    fill: { color: C.bgDark }, line: { type: "none" }, rectRadius: 0.08,
  });
  slide.addText("nnUNetv2_predict", {
    x: 0.95, y: 1.55, w: 11.5, h: 0.4,
    fontFace: "Consolas", fontSize: 18, bold: true, color: "FBBF24", margin: 0,
  });
  const cmdLines = [
    "  -i <input_dir>    -o <output_dir>",
    "  -d 100            -c 3d_fullres",
    "  -tr nnUNetTrainer -p nnUNetResEncUNetLPlans",
    "  -f 0 1 2 3 4      -chk checkpoint_best.pth",
    "  -step_size 0.5",
  ];
  slide.addText(
    cmdLines.map((l, i) => ({ text: l, options: { breakLine: i < cmdLines.length - 1 } })),
    {
      x: 0.95, y: 1.95, w: 11.5, h: 1.8,
      fontFace: "Consolas", fontSize: 14, color: "E2E8F0", margin: 0, valign: "top",
    }
  );

  // Three explanation cards below
  const cards = [
    { t: "Sliding window", b: "50% overlap (step_size 0.5) for smooth boundaries between patches.", c: C.primary },
    { t: "5-fold softmax", b: "Each fold yields a 6-class softmax volume. Averaged voxel-wise.", c: C.l3 },
    { t: "argmax → labels", b: "Final hard label from arg-max over the averaged probabilities.", c: C.l4 },
  ];
  cards.forEach((c, i) => {
    const x = 0.7 + i * 4.13;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 4.05, w: 3.93, h: 0.06,
      fill: { color: c.c }, line: { type: "none" },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 4.11, w: 3.93, h: 2.4,
      fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 },
    });
    slide.addText(c.t, {
      x: x + 0.2, y: 4.25, w: 3.55, h: 0.5,
      fontFace: "Calibri", fontSize: 16, bold: true, color: c.c, margin: 0,
    });
    slide.addText(c.b, {
      x: x + 0.2, y: 4.8, w: 3.55, h: 1.5,
      fontFace: "Calibri", fontSize: 12, color: C.ink, margin: 0, valign: "top",
    });
  });
})();

// ============================================================================
// SLIDE 10 — Why post-process
// ============================================================================

(function failureSlide() {
  slideNum = 10;
  const slide = newContentSlide("Why Post-Process? Failure Modes of Raw Output", "06 · Post-Processing", slideNum, TOTAL);

  slide.addText(
    "The network sees patches, not the whole spine. Three systematic failure modes appear in raw predictions:",
    {
      x: 0.7, y: 1.4, w: 12, h: 0.5, fontFace: "Calibri",
      fontSize: 14, color: C.ink, margin: 0, italic: true,
    }
  );

  const failures = [
    {
      h: "Mirror artefacts",
      s: "Contralateral blobs that look spine-like in patch context but lie outside the vertebral column.",
      c: C.l1,
    },
    {
      h: "Disconnected fragments",
      s: "Partial-volume noise on bone edges or low-contrast regions creates small islands.",
      c: C.l4,
    },
    {
      h: "Ordering violations",
      s: "Adjacent vertebrae like L2 and L3 get swapped on ambiguous boundaries.",
      c: C.l5,
    },
  ];
  failures.forEach((f, i) => {
    const x = 0.7 + i * 4.13;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 2.1, w: 3.93, h: 2.5,
      fill: { color: "FFFFFF" }, line: { color: f.c, width: 1.5 }, rectRadius: 0.12,
    });
    slide.addShape(pres.shapes.OVAL, {
      x: x + 0.25, y: 2.3, w: 0.5, h: 0.5,
      fill: { color: f.c }, line: { type: "none" },
    });
    slide.addText(`${i + 1}`, {
      x: x + 0.25, y: 2.3, w: 0.5, h: 0.5,
      fontFace: "Cambria", fontSize: 18, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(f.h, {
      x: x + 0.85, y: 2.3, w: 2.95, h: 0.5,
      fontFace: "Calibri", fontSize: 16, bold: true, color: f.c,
      valign: "middle", margin: 0,
    });
    slide.addText(f.s, {
      x: x + 0.25, y: 2.95, w: 3.55, h: 1.5,
      fontFace: "Calibri", fontSize: 12.5, color: C.ink,
      margin: 0, valign: "top",
    });
  });

  // Solution callout
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 5.0, w: 12.0, h: 1.5,
    fill: { color: "EBF4FA" }, line: { color: C.primary, width: 1.5 }, rectRadius: 0.12,
  });
  slide.addText("OUR SOLUTION", {
    x: 0.95, y: 5.15, w: 11.5, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  slide.addText(
    "Three deterministic, anatomically-informed rules applied in voxel space directly to the predicted label volume — no learned parameters, no thresholds to tune.",
    {
      x: 0.95, y: 5.5, w: 11.5, h: 0.95, fontFace: "Calibri",
      fontSize: 15, bold: true, color: C.ink, margin: 0, valign: "middle",
    }
  );
})();

// ============================================================================
// SLIDE 11 — Algorithm 1
// ============================================================================

(function algoSlide() {
  slideNum = 11;
  const slide = newContentSlide("Algorithm 1 — Anatomical Post-Processing", "06 · Post-Processing", slideNum, TOTAL);

  // Step strip on top
  const stepX0 = 0.7;
  const stepW = (12.0) / 3;
  const steps = [
    { n: "01", t: "Sagittal-strip corridor",   c: C.l1, formula: "[c_LR − 0.20·N_LR ,  c_LR + 0.20·N_LR]" },
    { n: "02", t: "Largest connected component", c: C.l3, formula: "26-conn., per label, ≥ 50 voxels" },
    { n: "03", t: "Vertical SI ordering",       c: C.l4, formula: "monotonic L1 → L5 SI centroids" },
  ];
  steps.forEach((s, i) => {
    const x = stepX0 + i * stepW;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.4, w: stepW - 0.15, h: 1.7,
      fill: { color: "FFFFFF" }, line: { color: s.c, width: 1.25 },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.4, w: stepW - 0.15, h: 0.06,
      fill: { color: s.c }, line: { type: "none" },
    });
    slide.addText(s.n, {
      x: x + 0.2, y: 1.55, w: 1.0, h: 0.5,
      fontFace: "Cambria", fontSize: 22, bold: true, color: s.c, margin: 0,
    });
    slide.addText(s.t, {
      x: x + 0.2, y: 2.05, w: stepW - 0.5, h: 0.45,
      fontFace: "Calibri", fontSize: 14, bold: true, color: C.ink, margin: 0,
    });
    slide.addText(s.formula, {
      x: x + 0.2, y: 2.5, w: stepW - 0.5, h: 0.5,
      fontFace: "Consolas", fontSize: 11, color: C.inkMute, italic: true, margin: 0,
    });
  });

  // Pseudocode block
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 3.3, w: 12.0, h: 3.2,
    fill: { color: C.bgDark }, line: { type: "none" }, rectRadius: 0.08,
  });
  slide.addText("PSEUDOCODE", {
    x: 0.95, y: 3.4, w: 11.5, h: 0.35,
    fontFace: "Calibri", fontSize: 10, bold: true, color: "8FB8DA", charSpacing: 4, margin: 0,
  });

  const pseudo = [
    { t: "1: ", c: "FBBF24" }, { t: "Identify axes:  LR = argmin·span(S>0,a)  ;  SI = argmax·span(S>0,a)", c: "E2E8F0", b: true },
    { t: "2: ", c: "FBBF24" }, { t: "Step 1 — sagittal strip:  S′[ |LR − c_LR| > 0.20·N_LR ] ← 0", c: "E2E8F0", b: true },
    { t: "3: ", c: "FBBF24" }, { t: "Step 2 — largest CC:  for ℓ ∈ {1,…,5}: keep arg max_C |C_ℓ|, drop if < 50", c: "E2E8F0", b: true },
    { t: "4: ", c: "FBBF24" }, { t: "Step 3 — SI ordering:  drop labels whose SI centroid breaks monotonicity", c: "E2E8F0", b: true },
    { t: "5: ", c: "FBBF24" }, { t: "return S′  (cleaned segmentation)", c: "E2E8F0" },
  ];
  // Build runs; manually break per logical line (every 2 entries)
  const runs = [];
  for (let i = 0; i < pseudo.length; i += 2) {
    runs.push({ text: pseudo[i].t, options: { color: pseudo[i].c, bold: true } });
    runs.push({
      text: pseudo[i + 1].t,
      options: { color: pseudo[i + 1].c, breakLine: i + 1 < pseudo.length - 1 },
    });
  }
  slide.addText(runs, {
    x: 0.95, y: 3.85, w: 11.5, h: 2.6,
    fontFace: "Consolas", fontSize: 13.5, paraSpaceAfter: 6, margin: 0, valign: "top",
  });
})();

// ============================================================================
// SLIDE 12 — Post-processing in action (image)
// ============================================================================

(function ppActionSlide() {
  slideNum = 12;
  const slide = newContentSlide("Post-Processing in Action (mid-sagittal slice)", "06 · Post-Processing", slideNum, TOTAL);

  slide.addImage({
    path: FIG("postprocess_steps.png"),
    x: 0.5, y: 1.35, w: 12.3, h: 4.2,
    sizing: { type: "contain", w: 12.3, h: 4.2 },
  });

  // Caption with three colored points
  const captions = [
    { c: C.l1, t: "Raw → Step 1:", s: "contralateral mirror blobs eliminated by the corridor constraint." },
    { c: C.l3, t: "Step 1 → Step 2:", s: "small islands (<50 voxels) discarded; one component retained per vertebra." },
    { c: C.l4, t: "Step 2 → Step 3:", s: "ordering enforcement removes any centroid breaking L1→L5 monotonicity." },
  ];
  captions.forEach((c, i) => {
    const y = 5.7 + i * 0.32;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.7, y: y + 0.08, w: 0.16, h: 0.16,
      fill: { color: c.c }, line: { type: "none" },
    });
    slide.addText([
      { text: c.t + " ", options: { bold: true, color: C.ink } },
      { text: c.s, options: { color: C.inkMute } },
    ], {
      x: 0.95, y: y, w: 12.0, h: 0.32,
      fontFace: "Calibri", fontSize: 12, valign: "middle", margin: 0,
    });
  });
})();

// ============================================================================
// SLIDE 13 — Mesh Extraction
// ============================================================================

(function meshSlide() {
  slideNum = 13;
  const slide = newContentSlide("Stage 4: Voxel → Surface (Marching Cubes + Taubin)", "07 · Mesh Extraction", slideNum, TOTAL);

  // Left — numbered steps
  const steps = [
    "Binary mask  M_ℓ = ( S = ℓ )  per retained label",
    "1-voxel zero pad (no boundary holes)",
    "Marching cubes at iso-level 0.5",
    "Voxel → NIfTI affine (RAS) → negate X,Y for LPS world coords",
    "Taubin smoothing (10 iterations) — volume-preserving low-pass",
    "Export STL (binary, 80-byte header)",
  ];
  steps.forEach((s, i) => {
    const y = 1.5 + i * 0.7;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.7, y: y + 0.08, w: 0.5, h: 0.5,
      fill: { color: C.primary }, line: { type: "none" },
    });
    slide.addText(`${i + 1}`, {
      x: 0.7, y: y + 0.08, w: 0.5, h: 0.5,
      fontFace: "Calibri", fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s, {
      x: 1.3, y: y, w: 5.5, h: 0.65,
      fontFace: "Calibri", fontSize: 13, color: C.ink, valign: "middle", margin: 0,
    });
  });

  // Right — the rendered 3D mesh
  slide.addImage({
    path: FIG("mesh_3d.png"),
    x: 7.2, y: 1.5, w: 5.6, h: 5.0,
    sizing: { type: "contain", w: 5.6, h: 5.0 },
  });
  slide.addText("Five smoothed surface meshes,\nLPS world coordinates", {
    x: 7.2, y: 6.55, w: 5.6, h: 0.5,
    fontFace: "Calibri", fontSize: 11, italic: true, color: C.inkMute,
    align: "center", valign: "middle", margin: 0,
  });
})();

// ============================================================================
// SLIDE 14 — Taubin Smoothing
// ============================================================================

(function taubinSlide() {
  slideNum = 14;
  const slide = newContentSlide("Taubin Smoothing in One Slide", "07 · Mesh Extraction", slideNum, TOTAL);

  // Top — comparison cards (Laplacian vs Taubin)
  const cards = [
    { h: "LAPLACIAN ONLY", s: "Iteratively smooths but shrinks the mesh — vertebrae get progressively smaller with every pass.",
      bg: "FEF2F2", border: C.l1, htext: C.l1 },
    { h: "TAUBIN (λ, μ)",  s: "Two alternating filters with positive λ and negative μ — smooths without volume loss.",
      bg: "ECFDF5", border: C.l2, htext: "166534" },
  ];
  cards.forEach((c, i) => {
    const x = 0.7 + i * 6.1;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 1.4, w: 5.9, h: 1.95,
      fill: { color: c.bg }, line: { color: c.border, width: 1.25 }, rectRadius: 0.1,
    });
    slide.addText(c.h, {
      x: x + 0.25, y: 1.55, w: 5.4, h: 0.4,
      fontFace: "Calibri", fontSize: 12, bold: true, color: c.htext, charSpacing: 4, margin: 0,
    });
    slide.addText(c.s, {
      x: x + 0.25, y: 2.0, w: 5.4, h: 1.3,
      fontFace: "Calibri", fontSize: 14, color: C.ink, margin: 0, valign: "top",
    });
  });

  // Math block
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 3.6, w: 12.0, h: 1.55,
    fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.1,
  });
  slide.addText("UPDATE RULE", {
    x: 0.95, y: 3.7, w: 11.5, h: 0.35,
    fontFace: "Calibri", fontSize: 10.5, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  slide.addText("v_(t+1)  =  ( I + μ · L ) ( I + λ · L )  ·  v_t", {
    x: 0.95, y: 4.05, w: 11.5, h: 0.7,
    fontFace: "Cambria", fontSize: 26, bold: true, color: C.primary,
    align: "center", valign: "middle", italic: true, margin: 0,
  });
  slide.addText(
    "L is the discrete Laplacian on mesh vertices. Choose 0 < λ and μ ≈ −λ/(1 + λ·k_PB) for a band-pass-like response.",
    {
      x: 0.95, y: 4.7, w: 11.5, h: 0.4,
      fontFace: "Calibri", fontSize: 11, italic: true, color: C.inkMute,
      align: "center", margin: 0,
    }
  );

  // Result
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 5.4, w: 12.0, h: 1.0,
    fill: { color: "EBF4FA" }, line: { color: C.primary, width: 1.5 }, rectRadius: 0.1,
  });
  slide.addText("RESULT", {
    x: 0.95, y: 5.5, w: 11.5, h: 0.3,
    fontFace: "Calibri", fontSize: 10.5, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  slide.addText(
    "Removes the marching-cubes staircase artefacts at the bone–background boundary  without  shrinking the vertebral body.",
    {
      x: 0.95, y: 5.85, w: 11.5, h: 0.55,
      fontFace: "Calibri", fontSize: 14, bold: true, color: C.ink,
      align: "center", valign: "middle", margin: 0,
    }
  );
})();

// ============================================================================
// SLIDE 15 — The Desktop App
// ============================================================================

(function appSlide() {
  slideNum = 15;
  const slide = newContentSlide("The Desktop Application", "08 · Application", slideNum, TOTAL);

  // Stack pills
  slide.addText("STACK", {
    x: 0.7, y: 1.4, w: 12, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });

  const stacks = [
    { name: "Streamlit",  desc: "browser-based UI", color: C.l1 },
    { name: "pywebview",  desc: "native macOS WKWebView wrapper", color: C.l3 },
    { name: "matplotlib", desc: "3-pane MPR", color: C.primary },
    { name: "Plotly Mesh3d", desc: "interactive 3D", color: C.l4 },
    { name: "trimesh",    desc: "marching cubes + Taubin", color: C.l5 },
    { name: "nnU-Net v2", desc: "model + predictor", color: C.secondary },
  ];
  stacks.forEach((s, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.7 + col * 4.13;
    const y = 1.85 + row * 1.0;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: y, w: 3.93, h: 0.85,
      fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.08,
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 0.08, h: 0.85,
      fill: { color: s.color }, line: { type: "none" },
    });
    slide.addText(s.name, {
      x: x + 0.25, y: y, w: 3.6, h: 0.42,
      fontFace: "Calibri", fontSize: 14, bold: true, color: C.ink,
      valign: "middle", margin: 0,
    });
    slide.addText(s.desc, {
      x: x + 0.25, y: y + 0.42, w: 3.6, h: 0.42,
      fontFace: "Calibri", fontSize: 11.5, italic: true, color: C.inkMute,
      valign: "middle", margin: 0,
    });
  });

  // Two launch modes
  slide.addText("LAUNCH MODES", {
    x: 0.7, y: 4.05, w: 12, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  const modes = [
    { h: "Native window (Finder)",
      s: "Double-click  launch_spine_seg_app.command  —  no browser, real Dock entry, traffic-light buttons.",
      c: C.primary },
    { h: "CLI",
      s: "python cli.py <input.nii.gz> -m nnUNet_results -o results/   —  scripts and batch jobs.",
      c: C.l4 },
  ];
  modes.forEach((m, i) => {
    const x = 0.7 + i * 6.1;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 4.45, w: 5.9, h: 1.4,
      fill: { color: "FFFFFF" }, line: { color: m.c, width: 1.25 }, rectRadius: 0.1,
    });
    slide.addText(m.h, {
      x: x + 0.25, y: 4.6, w: 5.4, h: 0.4,
      fontFace: "Calibri", fontSize: 15, bold: true, color: m.c, margin: 0,
    });
    slide.addText(m.s, {
      x: x + 0.25, y: 5.05, w: 5.4, h: 0.75,
      fontFace: "Calibri", fontSize: 12, color: C.ink, margin: 0, valign: "top",
    });
  });

  // File support callout
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 6.0, w: 12.0, h: 0.7,
    fill: { color: "FFF7ED" }, line: { color: C.l4, width: 1.0 }, rectRadius: 0.08,
  });
  slide.addText([
    { text: "FILE SUPPORT  ", options: { bold: true, color: C.l4, charSpacing: 4 } },
    { text: "single .nii / .nii.gz  or  a folder of cases · per-file upload limit ", options: { color: C.ink } },
    { text: "600 MB", options: { bold: true, color: C.l4 } },
    { text: "  · outputs saved to  ./results/<case_id>/", options: { color: C.ink } },
  ], {
    x: 0.95, y: 6.0, w: 11.5, h: 0.7,
    fontFace: "Calibri", fontSize: 12, valign: "middle", margin: 0,
  });
})();

// ============================================================================
// SLIDE 16 — App Architecture diagram
// ============================================================================

(function appArchSlide() {
  slideNum = 16;
  const slide = newContentSlide("App Architecture", "08 · Application", slideNum, TOTAL);

  // Top: native window
  flowBox(slide, 5.15,  1.45, 3.0, 0.85, "macOS native window\n(WKWebView)", "primary");
  arrow(slide, 6.65, 2.3, 6.65, 2.7);
  // Streamlit server
  flowBox(slide, 5.15,  2.7,  3.0, 0.85, "Streamlit server\n(localhost:rand)", "primary");
  arrow(slide, 6.65, 3.55, 6.65, 3.95);
  // run_pipeline
  flowBox(slide, 5.15,  3.95, 3.0, 0.85, "spine_seg.pipeline\nrun_pipeline()", "accent");
  // Branches
  arrow(slide, 5.15, 4.37, 4.55, 4.37);
  flowBox(slide, 1.55,  3.95, 3.0, 0.85, "nnUNetPredictor\n1- or 5-fold", "primary");
  arrow(slide, 8.15, 4.37, 8.75, 4.37);
  flowBox(slide, 8.75,  3.95, 3.0, 0.85, "3-step\npost-processing", "accent");
  arrow(slide, 6.65, 4.8, 6.65, 5.2);
  flowBox(slide, 5.15,  5.2,  3.0, 0.85, "Cleaned NIfTI\n+ 5 STL meshes", "output");
  arrow(slide, 8.15, 5.62, 8.75, 5.62);
  flowBox(slide, 8.75,  5.2,  3.0, 0.85, "Live MPR + 3D mesh\nin the same window", "output");

  // Caption
  slide.addText(
    "Single Python process; pywebview embeds a WKWebView pointed at the local Streamlit URL. Closing the window terminates the entire process group cleanly.",
    {
      x: 0.7, y: 6.4, w: 12, h: 0.5,
      fontFace: "Calibri", fontSize: 12, italic: true, color: C.inkMute,
      align: "center", margin: 0,
    }
  );
})();

// ============================================================================
// SLIDE 17 — Result: MPR overlay
// ============================================================================

(function mprResultSlide() {
  slideNum = 17;
  const slide = newContentSlide("Result: Multi-Planar View with Segmentation Overlay", "09 · Results", slideNum, TOTAL);

  slide.addImage({
    path: FIG("mpr_overlay.png"),
    x: 0.5, y: 1.35, w: 12.3, h: 4.6,
    sizing: { type: "contain", w: 12.3, h: 4.6 },
  });
  slide.addText(
    "Sliders in the app re-render these panels live. Bilinear interpolation on the MRI keeps the through-plane axis readable; nearest-neighbour on the labels keeps boundaries crisp.",
    {
      x: 0.7, y: 6.1, w: 12, h: 0.7,
      fontFace: "Calibri", fontSize: 12, italic: true, color: C.inkMute,
      align: "center", valign: "middle", margin: 0,
    }
  );
})();

// ============================================================================
// SLIDE 18 — Result: 3D mesh
// ============================================================================

(function mesh3dResultSlide() {
  slideNum = 18;
  const slide = newContentSlide("Result: 3D Surface Reconstruction", "09 · Results", slideNum, TOTAL);

  // Left — image
  slide.addImage({
    path: FIG("mesh_3d.png"),
    x: 0.5, y: 1.35, w: 6.5, h: 5.5,
    sizing: { type: "contain", w: 6.5, h: 5.5 },
  });

  // Right — three feature cards
  const features = [
    { h: "Fully interactive", s: "Drag to rotate · scroll to zoom · click legend to toggle individual vertebrae." },
    { h: "World-coordinate exact", s: "Each STL carries the original NIfTI affine transformed to LPS — opens correctly in 3D Slicer." },
    { h: "Surgeon-grade smoothness", s: "Marching cubes + Taubin removes voxel-staircase without shrinking the bone." },
  ];
  features.forEach((f, i) => {
    const y = 1.55 + i * 1.7;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 7.4, y: y, w: 5.4, h: 1.5,
      fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.1,
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 7.4, y: y, w: 0.08, h: 1.5,
      fill: { color: [C.l1, C.l3, C.l4][i] }, line: { type: "none" },
    });
    slide.addText(f.h, {
      x: 7.65, y: y + 0.15, w: 5.0, h: 0.45,
      fontFace: "Calibri", fontSize: 15, bold: true, color: C.ink, margin: 0,
    });
    slide.addText(f.s, {
      x: 7.65, y: y + 0.6, w: 5.0, h: 0.85,
      fontFace: "Calibri", fontSize: 12, color: C.inkMute, margin: 0, valign: "top",
    });
  });
})();

// ============================================================================
// SLIDE 19 — Voxel counts
// ============================================================================

(function voxelSlide() {
  slideNum = 19;
  const slide = newContentSlide("Quantitative — Per-Vertebra Voxel Counts", "09 · Results", slideNum, TOTAL);

  slide.addImage({
    path: FIG("voxel_counts.png"),
    x: 1.2, y: 1.4, w: 8.3, h: 4.7,
    sizing: { type: "contain", w: 8.3, h: 4.7 },
  });

  // Right column — interpretation panel
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 9.7, y: 1.4, w: 3.2, h: 4.7,
    fill: { color: "FFFFFF" }, line: { color: C.rule, width: 0.5 }, rectRadius: 0.1,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 9.7, y: 1.4, w: 3.2, h: 0.06,
    fill: { color: C.primary }, line: { type: "none" },
  });
  slide.addText("INTERPRETATION", {
    x: 9.85, y: 1.55, w: 2.95, h: 0.3,
    fontFace: "Calibri", fontSize: 10.5, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  slide.addText([
    { text: "All five lumbar bodies", options: { bold: true, color: C.ink, fontSize: 13, breakLine: true } },
    { text: "detected on this single ", options: { color: C.inkMute, fontSize: 12 } },
    { text: "test case.", options: { color: C.inkMute, fontSize: 12, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: "Counts decrease toward L5,", options: { color: C.ink, fontSize: 12, breakLine: true } },
    { text: "matching the smaller anatomical extent of the lower lumbar bodies in this subject.", options: { color: C.inkMute, fontSize: 12 } },
  ], {
    x: 9.85, y: 1.95, w: 2.95, h: 4.0, fontFace: "Calibri", margin: 0, valign: "top",
  });

  // Caption
  slide.addText("Cleaned segmentation, single test case (case 542 · sagittal T1 FLAIR).", {
    x: 0.7, y: 6.3, w: 12, h: 0.4,
    fontFace: "Calibri", fontSize: 11, italic: true, color: C.inkMute,
    align: "center", margin: 0,
  });
})();

// ============================================================================
// SLIDE 20 — Performance + Limitations (combined to keep deck tight)
// ============================================================================

(function perfLimitSlide() {
  slideNum = 20;
  const slide = newContentSlide("Performance & Limitations", "10 · Wrap-up", slideNum, TOTAL);

  // Performance table (left half)
  slide.addText("RUNTIME · M1 MAX (MPS)", {
    x: 0.7, y: 1.4, w: 6, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.primary, charSpacing: 4, margin: 0,
  });
  slide.addTable(
    [
      [
        { text: "Mode", options: { bold: true, color: "FFFFFF", fill: { color: C.primary } } },
        { text: "Time / case", options: { bold: true, color: "FFFFFF", fill: { color: C.primary } } },
      ],
      ["1-fold (default)",         "~12 min"],
      ["5-fold ensemble",          "~50–60 min"],
      ["1-fold CPU fallback",      "~25–35 min"],
      ["Post-processing",          "< 1 s"],
      ["Mesh extraction (5 STLs)", "~3–4 s"],
    ],
    {
      x: 0.7, y: 1.75, w: 6.0, colW: [3.3, 2.7],
      fontFace: "Calibri", fontSize: 13, color: C.ink,
      rowH: 0.4,
      border: { type: "solid", pt: 0.5, color: "EDF2F7" },
    }
  );
  slide.addText(
    "Reference RTX 5090 from the published pipeline: ~15 s (1 fold), ~60 s (5-fold). MPS on M1 Max is ≈ 50× slower — acceptable for a single-user desktop app.",
    {
      x: 0.7, y: 4.6, w: 6.0, h: 0.95,
      fontFace: "Calibri", fontSize: 11, italic: true, color: C.inkMute, margin: 0, valign: "top",
    }
  );

  // Limitations (right half)
  slide.addText("LIMITATIONS", {
    x: 7.0, y: 1.4, w: 6, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: C.l4, charSpacing: 4, margin: 0,
  });
  const lims = [
    { h: "T1-FLAIR-only model", s: "T2 / STIR will produce poor segmentations until re-trained." },
    { h: "Standard-anatomy assumption", s: "Transitional vertebrae (lumbarised S1, sacralised L5) may legitimately lose a label." },
    { h: "Apple-Silicon latency", s: "5-fold ensemble too slow for true real-time; CUDA collapses it to seconds." },
    { h: "Voxel-space rules", s: "LR/SI axes inferred from labeled bbox — works but ignores the affine orientation." },
  ];
  lims.forEach((l, i) => {
    const y = 1.78 + i * 1.18;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 7.0, y: y, w: 0.06, h: 1.05,
      fill: { color: C.l4 }, line: { type: "none" },
    });
    slide.addText(l.h, {
      x: 7.18, y: y, w: 5.6, h: 0.4,
      fontFace: "Calibri", fontSize: 13.5, bold: true, color: C.ink, valign: "middle", margin: 0,
    });
    slide.addText(l.s, {
      x: 7.18, y: y + 0.4, w: 5.6, h: 0.65,
      fontFace: "Calibri", fontSize: 11.5, color: C.inkMute, italic: true, valign: "top", margin: 0,
    });
  });

  // Spider chart-like callout — actually keeping it textual
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.7, y: 5.8, w: 12.0, h: 1.0,
    fill: { color: "EBF4FA" }, line: { color: C.primary, width: 1.5 }, rectRadius: 0.1,
  });
  slide.addText([
    { text: "Bottom line.  ", options: { bold: true, color: C.primary, fontSize: 14 } },
    { text: "Three deterministic image-processing rules eliminate the dominant failure modes of a purely data-driven model — and the whole pipeline runs from a single click on Apple Silicon.", options: { color: C.ink, fontSize: 13, italic: true } },
  ], {
    x: 0.95, y: 5.8, w: 11.5, h: 1.0, fontFace: "Calibri", margin: 0, valign: "middle",
  });
})();

// ============================================================================
// SLIDE 21 — Conclusion / Thank you
// ============================================================================

(function thankYouSlide() {
  slideNum = 21;
  const slide = pres.addSlide();
  slide.background = { color: C.bgDark };

  // Top stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.18, fill: { color: C.primary }, line: { type: "none" },
  });
  // Bottom stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.18, w: W, h: 0.18, fill: { color: C.primary }, line: { type: "none" },
  });
  // Vertebra dots in top-right (motif repeat)
  [C.l1, C.l2, C.l3, C.l4, C.l5].forEach((c, i) => {
    slide.addShape(pres.shapes.OVAL, {
      x: W - 2.6 + i * 0.45, y: 0.45, w: 0.32, h: 0.32,
      fill: { color: c }, line: { type: "none" },
    });
  });

  slide.addText("FROM A SINGLE NIfTI VOLUME TO", {
    x: 0.7, y: 1.7, w: 12, h: 0.5,
    fontFace: "Calibri", fontSize: 14, bold: true, color: "8FB8DA",
    charSpacing: 6, margin: 0,
  });
  slide.addText("clinical-grade segmentation,", {
    x: 0.7, y: 2.25, w: 12, h: 1.0,
    fontFace: "Cambria", fontSize: 44, bold: true, color: "FFFFFF", margin: 0,
  });
  slide.addText("in one click.", {
    x: 0.7, y: 3.15, w: 12, h: 1.0,
    fontFace: "Cambria", fontSize: 44, bold: true, italic: true, color: C.l3, margin: 0,
  });

  // Conclusion bullet block
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 4.45, w: 0.06, h: 1.65,
    fill: { color: C.l3 }, line: { type: "none" },
  });
  slide.addText([
    { text: "Raw T1 FLAIR  →  nnU-Net inference  →  3-step post-processing  →  NIfTI + STL  →  interactive desktop app.",
      options: { color: "E2ECF7", fontSize: 14, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Three deterministic rules (sagittal corridor, largest CC, SI ordering) eliminate the dominant failure modes of a purely data-driven model.",
      options: { color: "B0C7DC", fontSize: 13, italic: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Marching cubes + Taubin produce surface meshes that register correctly with the source MRI in 3D Slicer (LPS).",
      options: { color: "B0C7DC", fontSize: 13, italic: true } },
  ], {
    x: 0.95, y: 4.45, w: 11.7, h: 1.85, fontFace: "Calibri", margin: 0, valign: "top",
  });

  // Thank-you
  slide.addText("Thank You", {
    x: 0.7, y: 6.4, w: 6, h: 0.6,
    fontFace: "Cambria", fontSize: 28, bold: true, color: "FFFFFF", margin: 0,
  });
  slide.addText("Questions?", {
    x: 0.7, y: 7.0, w: 6, h: 0.35,
    fontFace: "Calibri", fontSize: 14, italic: true, color: "8FB8DA", margin: 0,
  });

  slide.addText("Varun Dasoju  ·  Image Processing  ·  Term Project", {
    x: W - 6.5, y: 6.95, w: 6.0, h: 0.35,
    fontFace: "Calibri", fontSize: 11, color: "8FB8DA", italic: true,
    align: "right", margin: 0,
  });
})();

// ----------------------------------------------------------------------------
// Write file
// ----------------------------------------------------------------------------

pres.writeFile({ fileName: path.join(__dirname, "spine_seg_presentation.pptx") })
  .then((file) => console.log("Wrote", file))
  .catch((err) => { console.error("Failed:", err); process.exit(1); });
