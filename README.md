# YOLO-World Evaluation Framework

A prompt-driven evaluation framework for assessing [YOLO-World](https://github.com/AILab-CVC/YOLO-World) open-vocabulary object detection. The framework measures how well the model detects target objects, resists semantically confusable labels, and avoids hallucinating distractor objects across diverse scene types.

## Overview

This project evaluates the **YOLO-World v2** model (`yolov8s-worldv2`) on a curated subset of **103 images** from the COCO val2017 dataset, spanning **713 ground-truth annotations** across **64 object classes**. The pipeline has two stages:

1. **Inference** (`yoloworld-usage.py`) -- runs YOLO-World with per-image text prompts and exports structured predictions.
2. **Evaluation** (`evaluate.py`) -- compares predictions against COCO-format ground truth and produces COCO-style detection metrics, failure analysis, and hallucination diagnostics.

### Prompt Design

Each image is assigned a prompt composed of three label groups defined in `prompts/prompts.csv`:

| Label Group | Description |
|---|---|
| **Targets** | Object classes that genuinely appear in the image (aligned with GT annotations). |
| **Confusables** | Semantically similar classes that do *not* appear (e.g., "mug" when the GT is "cup"). |
| **Distractors** | Unrelated classes absent from the image, used to measure hallucination. |

### Scene Types

Images are categorized into three scene types to enable stratified analysis:

| Scene Type | Images | GT Boxes | Description |
|---|---|---|---|
| Ambiguous | 34 | 314 | Scenes where visual similarity between objects is high. |
| Natural | 34 | 235 | Typical everyday scenes. |
| Unusual | 35 | 164 | Rare or atypical compositions. |

## Evaluation Design

### Metrics

The evaluation computes four categories of metrics.

#### 1. COCO-Style Detection Metrics

Standard metrics computed using IoU thresholds from 0.50 to 0.95 (step 0.05), with up to 100 detections per image:

| Metric | Description |
|---|---|
| **AP@50** | Average Precision at IoU threshold 0.50, averaged across all target classes. |
| **AP@75** | Average Precision at the stricter IoU threshold 0.75. |
| **mAP@[.50:.95]** | Mean AP averaged over 10 IoU thresholds (0.50, 0.55, ..., 0.95). The primary detection quality metric. |
| **Recall@50** | Fraction of GT boxes matched by at least one target prediction at IoU >= 0.50. |
| **AR@100** | Average Recall with up to 100 detections, averaged over all IoU thresholds. |
| **Recall (S/M/L)** | Recall@50 stratified by object size: small (area < 32x32), medium (32x32 to 96x96), large (> 96x96). |

#### 2. Failure Analysis Metrics

Each unmatched GT box is classified into one of five failure modes:

| Failure Mode | Description |
|---|---|
| **Miss** | No prediction overlaps the GT box above the localization IoU threshold. |
| **Semantic Confusion** | The best-overlapping prediction uses a *confusable* label instead of the correct target label. |
| **Wrong Class** | The best-overlapping prediction uses a label that is neither the correct target nor a confusable. |
| **Localization Error** | A correct-class prediction exists but its IoU with the GT box is below the match threshold (0.5) while still above the localization minimum (0.1). |
| **Duplicate** | A correct-class prediction overlaps a GT box that was already matched by a higher-confidence prediction. |

Rates are computed as count / total GT boxes.

#### 3. Hallucination Metrics

False positive predictions whose maximum IoU with any GT box is below 0.1 are counted as hallucinations. They are categorized by which label group the predicted class belongs to:

| Metric | Description |
|---|---|
| **Hallucination FP/image (total)** | Total hallucinated false positives per image. |
| **Hallucination FP/image (target)** | Hallucinations predicting a *target* class label. |
| **Hallucination FP/image (confusable)** | Hallucinations predicting a *confusable* class label. |
| **Hallucination FP/image (distractor)** | Hallucinations predicting a *distractor* class label. |

#### 4. Image-Level Presence Metrics

Binary per-image indicators aggregated across the dataset:

| Metric | Description |
|---|---|
| **Target Accuracy** | Fraction of images where at least one target class is detected. |
| **Confusion Rate** | Fraction of images where at least one confusable class is detected. |
| **Hallucination Rate** | Fraction of images where at least one distractor class is detected. |

## Results

All experiments use `yolov8s-worldv2.pt` with image size 640 on the same 103-image dataset. Two confidence thresholds are compared.

### Overall Performance

| Metric | conf = 0.25 | conf = 0.001 |
|---|---|---|
| AP@50 | 0.5721 | 0.7233 |
| AP@75 | 0.4987 | 0.6217 |
| mAP@[.50:.95] | 0.4606 | 0.5593 |
| Recall@50 | 0.5694 | 0.8205 |
| AR@100 | 0.4583 | 0.6187 |
| Recall (small) | 0.2797 | 0.6356 |
| Recall (medium) | 0.6250 | 0.9129 |
| Recall (large) | 0.8216 | 0.9108 |
| Miss Rate | 0.3184 | 0.0168 |
| Semantic Confusion Rate | 0.0309 | 0.0519 |
| Wrong-Class Rate | 0.0337 | 0.0870 |
| Localization Error Rate | 0.0589 | 0.0701 |
| Duplicate Rate | 0.0224 | 0.7770 |
| Hallucination FP/image | 0.9903 | 65.8932 |
| Image-Level Target Accuracy | 99.03% | 99.03% |
| Image-Level Confusion Rate | 27.18% | 85.44% |
| Image-Level Hallucination Rate | 31.07% | 93.20% |

**Key takeaway:** Lowering the confidence threshold from 0.25 to 0.001 substantially improves recall and mAP (0.46 to 0.56) but introduces massive hallucination (0.99 to 65.89 FP/image) and duplicates (2.2% to 77.7%).

### Performance by Scene Type (conf = 0.25)

| Metric | Ambiguous | Natural | Unusual |
|---|---|---|---|
| AP@50 | 0.6339 | 0.5332 | 0.5835 |
| mAP@[.50:.95] | 0.5231 | 0.4302 | 0.4430 |
| Recall@50 | 0.5860 | 0.5362 | 0.5854 |
| Miss Rate | 0.2994 | 0.3830 | 0.2622 |
| Semantic Confusion Rate | 0.0414 | 0.0170 | 0.0305 |
| Localization Error Rate | 0.0414 | 0.0468 | 0.1098 |
| Hallucination FP/image | 1.2059 | 0.9706 | 0.8000 |

### Performance by Scene Type (conf = 0.001)

| Metric | Ambiguous | Natural | Unusual |
|---|---|---|---|
| AP@50 | 0.7442 | 0.6728 | 0.7362 |
| mAP@[.50:.95] | 0.5949 | 0.5185 | 0.5468 |
| Recall@50 | 0.8471 | 0.7702 | 0.8415 |
| Miss Rate | 0.0159 | 0.0255 | 0.0061 |
| Semantic Confusion Rate | 0.0669 | 0.0383 | 0.0427 |
| Localization Error Rate | 0.0382 | 0.1149 | 0.0671 |
| Hallucination FP/image | 90.0294 | 61.7941 | 46.4286 |

## Usage

### Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics) (provides the YOLO-World implementation)

Install dependencies:

```bash
pip install torch ultralytics
```

The YOLO-World weights file (`yolov8s-worldv2.pt`) will be downloaded automatically by Ultralytics on the first run, or you can place it in the project root manually.

### Step 1: Run Inference

`yoloworld-usage.py` loads per-image prompts from a CSV, runs YOLO-World inference, and writes predictions to JSON along with annotated visualization images.

```bash
python yoloworld-usage.py \
    --csv prompts/prompts.csv \
    --images-dir images/ \
    --conf 0.25 \
    --outdir outputs_conf25
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to the prompts CSV file. |
| `--images-dir` | *(required)* | Root directory containing images (searched recursively). |
| `--weights` | `yolov8s-worldv2.pt` | Path to YOLO-World model weights. |
| `--conf` | `0.001` | Confidence threshold for detections. |
| `--imgsz` | `640` | Inference image size (pixels). |
| `--device` | auto | Device string (`mps`, `cpu`, `0`). Auto-selects MPS on Apple Silicon. |
| `--max` | `0` (no limit) | Maximum number of images to process. |
| `--outdir` | `outputs` | Output directory. |

**Outputs:**

- `<outdir>/predictions.json` -- structured prediction results for all images.
- `<outdir>/vis/` -- annotated images with bounding box overlays.

To reproduce the two configurations used in this evaluation:

```bash
# conf = 0.25 (fewer false positives)
python yoloworld-usage.py \
    --csv prompts/prompts.csv \
    --images-dir images/ \
    --conf 0.25 \
    --outdir outputs_conf25

# conf = 0.001 (higher recall, more noise)
python yoloworld-usage.py \
    --csv prompts/prompts.csv \
    --images-dir images/ \
    --conf 0.001 \
    --outdir outputs_conf001
```

### Step 2: Run Evaluation

`evaluate.py` loads the predictions JSON, COCO-format ground truth, and the prompts CSV, then computes all detection and failure-analysis metrics.

```bash
python evaluate.py \
    --predictions outputs_conf25/predictions.json \
    --gt-json images/instances_val2017_select.json \
    --prompts-csv prompts/prompts.csv \
    --outdir outputs_conf25/eval_box
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--predictions` | `outputs_conf25/predictions.json` | Path to predictions JSON from Step 1. |
| `--gt-json` | `images/instances_val2017_select.json` | Path to COCO-style ground-truth annotations. |
| `--prompts-csv` | `prompts/prompts.csv` | Path to prompts CSV. |
| `--outdir` | `outputs_conf25/eval_box` | Directory for evaluation output files. |
| `--max-dets` | `100` | Max target detections per image for standard metrics. |
| `--localization-min-iou` | `0.1` | Minimum IoU to classify a same-class miss as localization error. |
| `--hallucination-max-iou` | `0.1` | Predictions with max GT IoU below this are counted as hallucinations. |

**Outputs:**

- `metrics_summary.json` -- full metrics broken down by overall and per-scene-type.
- `per_image_failures.csv` -- per-image counts of TPs, FNs, and each failure/hallucination type.
- `confusion_pairs.csv` -- aggregated (GT label, predicted label) pairs for semantic confusion and wrong-class errors.

### Running Tests

```bash
python -m unittest discover -s tests
```

## Project Structure

```
YOLO_Evaluation/
├── yoloworld-usage.py              # Inference script
├── evaluate.py                     # Evaluation script
├── prompts/
│   └── prompts.csv                 # Per-image prompts (targets, confusables, distractors)
├── images/
│   ├── instances_val2017_select.json   # COCO-format GT annotations (103 images)
│   └── ...                         # Image files
├── outputs_conf25/                 # Results with conf=0.25
│   ├── predictions.json
│   ├── vis/                        # Annotated images
│   └── eval_box/
│       ├── metrics_summary.json
│       ├── per_image_failures.csv
│       └── confusion_pairs.csv
├── outputs_conf001/                # Results with conf=0.001
│   ├── predictions.json
│   └── eval_box_conf001/
│       ├── metrics_summary.json
│       ├── per_image_failures.csv
│       └── confusion_pairs.csv
└── tests/
    └── test_evaluate.py            # Unit and smoke tests
```
