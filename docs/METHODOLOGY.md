# Methodology

## Pipeline

1. Extract the green channel, normalize intensity, apply CLAHE (clip limit 2.0, 8 × 8 tile grid), median filter (3 × 3), and mask the field of view.
2. Enhance vessels using multiscale morphological black-hat responses (kernel sizes 7, 11, 15).
3. Form percentile-based candidate and seed maps. A06 ranks candidate locations using a weighted fuzzy-support, Harris-response and vesselness score (0.35, 0.40, 0.25). The reference fuzzy support combines local similarity and inverse normalized gradient magnitude with weights 0.55 and 0.45.
4. Select spaced seed points, target 77 for A06, and dilate with radius 1. L00 instead uses percentile polling; its configured target of 35 is inactive, so its actual count varies.
5. Grow regions. A06 uses edge-delayed traversal, 8-connectivity, Kang-inspired product priority, connected-seed-component region means and reconsideration of rejected candidates from another accepted parent. Accept a candidate when its local vesselness difference is at most 18 OR its region-mean difference is at most 36 (2 × 18). L00 uses BFS with local-path acceptance.
6. Remove small connected components and fill small holes (configured area thresholds 8); respect the FoV and 500,000 iteration cap.

Exact implementation and frozen YAML take precedence over this summary. This is an adapted APSRG pipeline; it uses a different enhancement path than the original IUWT-based work.

## Controlled ablation

L00 is contextual. A00–A07 form the controlled comparison series:

| ID | Configuration |
|---|---|
| A00 | Vesselness-only seeds, target 35, no dilation |
| A01 | Selective Fuzzy–Harris seeds, target 35 |
| A02 | Target 77 |
| A03 | Radius-1 seed dilation |
| A04 | Edge-delayed traversal, local-only acceptance |
| A05 | Region-only acceptance |
| A06 | Local-or-region acceptance (primary) |
| A07 | Hybrid priority control |

Other settings are maintained as defined by the frozen configurations. Traversal changes can also change the effective ordering/priority; consult the archived contrast validation table rather than treating all algorithm effects as independent.

## Evaluation

All labeled evaluations use the 20 DRIVE development images (IDs 21–40) within the FoV. Pixel-level confusion counts yield accuracy, precision, recall, specificity, F1, IoU and balanced accuracy; macro summaries average per-image values. F1 = 2TP/(2TP+FP+FN); IoU = TP/(TP+FP+FN); recall = TP/(TP+FN). See metrics.py for zero-denominator handling.

The two matched primary contrasts are A06 versus A04 and A06 versus A05, on F1 and IoU. Use two-sided Wilcoxon signed-rank tests, Holm adjustment across four primary hypotheses, rank-biserial effect sizes, and percentile paired bootstrap intervals (10,000 resamples, seed 42). Other comparisons are exploratory; the original eight-contrast within-metric Holm results are also retained. Inference is conditional on the development sample.

## Sensitivity

The thirteen settings comprise S00 (A06 reference), four fuzzy-support alternatives, four seed-score weight vectors and four regional tolerance multipliers. The scoped adapter changes fuzzy-support weights for the applicable runs without rewriting the core segmentation module. Preserve the reference literal 0.45 coefficient. No retuning or selecting a replacement primary configuration.

## Timing and descriptive test predictions

Archived CPU measurements report six fixed DRIVE development IDs, five repetitions and two reported methods, extracted from a broader schedule. The timer includes preprocessing and segmentation, excludes decoding and evaluation, and uses one OpenCV thread with OpenCL disabled. A new two-method run is not the same timing experiment; see RUNTIME_PROTOCOL.json.

Test IDs 1–20 are for local prediction and descriptive mask disagreement only. No official test accuracy, independent generalization or clinical performance is established.
