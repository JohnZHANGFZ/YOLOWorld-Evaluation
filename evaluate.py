from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
COCO_SMALL_AREA = 32 * 32
COCO_MEDIUM_AREA = 96 * 96


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _xywh_to_xyxy(bbox_xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return float(x), float(y), float(x + w), float(y + h)


def _compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _size_bucket(area: float) -> str:
    if area < COCO_SMALL_AREA:
        return "small"
    if area < COCO_MEDIUM_AREA:
        return "medium"
    return "large"


@dataclass(frozen=True)
class PromptEntry:
    image_id: str
    scene_type: str
    targets: frozenset[str]
    confusables: frozenset[str]
    distractors: frozenset[str]

    @property
    def prompt_labels(self) -> Set[str]:
        return set(self.targets | self.confusables | self.distractors)

    def label_group(self, label: str) -> str:
        if label in self.targets:
            return "target"
        if label in self.confusables:
            return "confusable"
        if label in self.distractors:
            return "distractor"
        return "unknown"


@dataclass(frozen=True)
class ImageInfo:
    image_id: str
    coco_image_id: int
    width: int
    height: int


@dataclass(frozen=True)
class GroundTruthBox:
    annotation_id: int
    image_id: str
    category_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    area: float
    size_bucket: str


@dataclass(frozen=True)
class PredictionBox:
    pred_id: str
    image_id: str
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass(frozen=True)
class GTFailureAssignment:
    category: str
    gt_label: str
    predicted_label: Optional[str]
    best_iou: float
    best_same_class_iou: float


@dataclass
class EvaluationArtifacts:
    summary: Dict[str, Any]
    per_image_rows: List[Dict[str, Any]]
    confusion_rows: List[Dict[str, Any]]


def _split_csv_labels(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    labels = {_normalize_label(item) for item in value.split(";") if _normalize_label(item)}
    return frozenset(labels)


def load_prompts(csv_path: str) -> Dict[str, PromptEntry]:
    prompts: Dict[str, PromptEntry] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"image_id", "scene_type", "targets", "confusables", "distractors"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Prompts CSV is missing required columns: {missing}")

        for row in reader:
            image_id = row["image_id"].strip()
            if not image_id:
                continue
            prompts[image_id] = PromptEntry(
                image_id=image_id,
                scene_type=row["scene_type"].strip(),
                targets=_split_csv_labels(row["targets"]),
                confusables=_split_csv_labels(row["confusables"]),
                distractors=_split_csv_labels(row["distractors"]),
            )
    return prompts


def load_ground_truth(gt_json_path: str) -> Tuple[Dict[str, ImageInfo], Dict[str, List[GroundTruthBox]]]:
    with open(gt_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    category_lookup = {item["id"]: _normalize_label(item["name"]) for item in data["categories"]}
    images_by_coco_id = {
        item["id"]: ImageInfo(
            image_id=item["file_name"],
            coco_image_id=item["id"],
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in data["images"]
    }

    gt_by_image: Dict[str, List[GroundTruthBox]] = defaultdict(list)
    for annotation in data["annotations"]:
        image_info = images_by_coco_id[annotation["image_id"]]
        bbox_xyxy = _xywh_to_xyxy(annotation["bbox"])
        area = float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]))
        gt_by_image[image_info.image_id].append(
            GroundTruthBox(
                annotation_id=int(annotation["id"]),
                image_id=image_info.image_id,
                category_name=category_lookup[annotation["category_id"]],
                bbox_xyxy=bbox_xyxy,
                area=area,
                size_bucket=_size_bucket(area),
            )
        )

    image_info_by_name = {info.image_id: info for info in images_by_coco_id.values()}
    for image_id in image_info_by_name:
        gt_by_image.setdefault(image_id, [])

    return image_info_by_name, gt_by_image


def load_predictions(predictions_json_path: str) -> Tuple[Dict[str, Any], Dict[str, List[PredictionBox]]]:
    with open(predictions_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or "predictions" not in data:
        raise ValueError("Predictions JSON must contain a top-level 'predictions' field.")

    predictions_by_image: Dict[str, List[PredictionBox]] = defaultdict(list)
    counters: Counter[str] = Counter()
    for entry in data.get("predictions", []):
        image_id = os.path.basename(entry["image"])
        for det in entry.get("detections", []):
            pred_index = counters[image_id]
            counters[image_id] += 1
            bbox = det.get("bbox_xyxy") or []
            if len(bbox) != 4:
                continue
            predictions_by_image[image_id].append(
                PredictionBox(
                    pred_id=f"{image_id}#{pred_index}",
                    image_id=image_id,
                    class_name=_normalize_label(str(det.get("class_name", ""))),
                    confidence=float(det.get("confidence") or 0.0),
                    bbox_xyxy=tuple(float(v) for v in bbox),
                )
            )

    for image_id, image_predictions in predictions_by_image.items():
        predictions_by_image[image_id] = sorted(
            image_predictions,
            key=lambda pred: pred.confidence,
            reverse=True,
        )

    metadata = {key: value for key, value in data.items() if key != "predictions"}
    return metadata, predictions_by_image


def _select_target_predictions(
    prompts: Dict[str, PromptEntry],
    predictions_by_image: Dict[str, List[PredictionBox]],
    image_ids: Sequence[str],
    max_dets: int,
) -> Dict[str, List[PredictionBox]]:
    selected: Dict[str, List[PredictionBox]] = {}
    for image_id in image_ids:
        prompt = prompts[image_id]
        image_predictions = [
            pred
            for pred in predictions_by_image.get(image_id, [])
            if pred.class_name in prompt.targets
        ]
        selected[image_id] = image_predictions[:max_dets]
    return selected


def _prepare_classwise_inputs(
    prompts: Dict[str, PromptEntry],
    image_ids: Sequence[str],
    gt_by_image: Dict[str, List[GroundTruthBox]],
    target_predictions_by_image: Dict[str, List[PredictionBox]],
) -> Tuple[Dict[str, Dict[str, List[GroundTruthBox]]], Dict[str, List[PredictionBox]], int]:
    gt_by_class: Dict[str, Dict[str, List[GroundTruthBox]]] = defaultdict(lambda: defaultdict(list))
    preds_by_class: Dict[str, List[PredictionBox]] = defaultdict(list)
    total_gt = 0

    for image_id in image_ids:
        prompt = prompts[image_id]
        for gt in gt_by_image.get(image_id, []):
            if gt.category_name not in prompt.targets:
                continue
            gt_by_class[gt.category_name][image_id].append(gt)
            total_gt += 1
        for pred in target_predictions_by_image.get(image_id, []):
            preds_by_class[pred.class_name].append(pred)

    for class_name in preds_by_class:
        preds_by_class[class_name].sort(key=lambda pred: pred.confidence, reverse=True)

    return gt_by_class, preds_by_class, total_gt


def _compute_ap(tp_flags: Sequence[int], fp_flags: Sequence[int], num_gt: int) -> Optional[float]:
    if num_gt == 0:
        return None
    if not tp_flags:
        return 0.0

    tp_cumulative: List[int] = []
    fp_cumulative: List[int] = []
    tp_total = 0
    fp_total = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp_total += tp_flag
        fp_total += fp_flag
        tp_cumulative.append(tp_total)
        fp_cumulative.append(fp_total)

    precisions = []
    recalls = []
    for tp_value, fp_value in zip(tp_cumulative, fp_cumulative):
        precisions.append(_safe_div(tp_value, tp_value + fp_value))
        recalls.append(_safe_div(tp_value, num_gt))

    ap = 0.0
    for recall_threshold in range(101):
        target_recall = recall_threshold / 100.0
        precision_candidates = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= target_recall
        ]
        ap += max(precision_candidates) if precision_candidates else 0.0
    return ap / 101.0


def _match_class_predictions(
    predictions: Sequence[PredictionBox],
    gt_by_image: Dict[str, List[GroundTruthBox]],
    iou_threshold: float,
) -> Dict[str, Any]:
    matched_gt_ids: Set[int] = set()
    matched_pred_ids: Set[str] = set()
    duplicate_pred_ids: Set[str] = set()
    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for pred in predictions:
        best_any_iou = 0.0
        best_unmatched_iou = 0.0
        best_unmatched_gt: Optional[GroundTruthBox] = None
        for gt in gt_by_image.get(pred.image_id, []):
            iou = _compute_iou(pred.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_any_iou:
                best_any_iou = iou
            if gt.annotation_id in matched_gt_ids:
                continue
            if iou > best_unmatched_iou:
                best_unmatched_iou = iou
                best_unmatched_gt = gt

        if best_unmatched_gt is not None and best_unmatched_iou >= iou_threshold:
            matched_gt_ids.add(best_unmatched_gt.annotation_id)
            matched_pred_ids.add(pred.pred_id)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
            if best_any_iou >= iou_threshold:
                duplicate_pred_ids.add(pred.pred_id)

    return {
        "tp_flags": tp_flags,
        "fp_flags": fp_flags,
        "matched_gt_ids": matched_gt_ids,
        "matched_pred_ids": matched_pred_ids,
        "duplicate_pred_ids": duplicate_pred_ids,
    }


def _compute_detection_metrics(
    prompts: Dict[str, PromptEntry],
    gt_by_image: Dict[str, List[GroundTruthBox]],
    predictions_by_image: Dict[str, List[PredictionBox]],
    image_ids: Sequence[str],
    iou_thresholds: Sequence[float],
    max_dets: int,
) -> Dict[str, Any]:
    target_predictions_by_image = _select_target_predictions(
        prompts=prompts,
        predictions_by_image=predictions_by_image,
        image_ids=image_ids,
        max_dets=max_dets,
    )
    gt_by_class, preds_by_class, total_gt = _prepare_classwise_inputs(
        prompts=prompts,
        image_ids=image_ids,
        gt_by_image=gt_by_image,
        target_predictions_by_image=target_predictions_by_image,
    )

    threshold_results: Dict[float, Dict[str, Any]] = {}
    class_names = sorted(gt_by_class)
    for iou_threshold in iou_thresholds:
        class_ap_values: List[float] = []
        matched_gt_ids: Set[int] = set()
        matched_pred_ids: Set[str] = set()
        duplicate_pred_ids: Set[str] = set()

        for class_name in class_names:
            class_predictions = preds_by_class.get(class_name, [])
            class_gt_by_image = gt_by_class[class_name]
            num_gt = sum(len(items) for items in class_gt_by_image.values())
            match_result = _match_class_predictions(
                predictions=class_predictions,
                gt_by_image=class_gt_by_image,
                iou_threshold=iou_threshold,
            )
            class_ap = _compute_ap(
                match_result["tp_flags"],
                match_result["fp_flags"],
                num_gt,
            )
            if class_ap is not None:
                class_ap_values.append(class_ap)

            matched_gt_ids.update(match_result["matched_gt_ids"])
            matched_pred_ids.update(match_result["matched_pred_ids"])
            duplicate_pred_ids.update(match_result["duplicate_pred_ids"])

        threshold_results[iou_threshold] = {
            "mean_ap": _mean(class_ap_values),
            "recall": _safe_div(len(matched_gt_ids), total_gt),
            "matched_gt_ids": matched_gt_ids,
            "matched_pred_ids": matched_pred_ids,
            "duplicate_pred_ids": duplicate_pred_ids,
        }

    gt_by_annotation_id = {
        gt.annotation_id: gt
        for image_id in image_ids
        for gt in gt_by_image.get(image_id, [])
        if gt.category_name in prompts[image_id].targets
    }
    matched_gt_50 = threshold_results[0.5]["matched_gt_ids"] if 0.5 in threshold_results else set()
    size_totals: Counter[str] = Counter(gt.size_bucket for gt in gt_by_annotation_id.values())
    size_matches: Counter[str] = Counter(
        gt_by_annotation_id[annotation_id].size_bucket
        for annotation_id in matched_gt_50
        if annotation_id in gt_by_annotation_id
    )

    return {
        "num_target_predictions": sum(len(items) for items in target_predictions_by_image.values()),
        "num_gt": total_gt,
        "num_classes": len(class_names),
        "ap50": threshold_results[0.5]["mean_ap"],
        "ap75": threshold_results[0.75]["mean_ap"],
        "map50_95": _mean(result["mean_ap"] for result in threshold_results.values()),
        "recall50": threshold_results[0.5]["recall"],
        "ar100": _mean(result["recall"] for result in threshold_results.values()),
        "recall_small": _safe_div(size_matches["small"], size_totals["small"]) if size_totals["small"] else None,
        "recall_medium": _safe_div(size_matches["medium"], size_totals["medium"]) if size_totals["medium"] else None,
        "recall_large": _safe_div(size_matches["large"], size_totals["large"]) if size_totals["large"] else None,
        "tp50": len(matched_gt_50),
        "fn50": total_gt - len(matched_gt_50),
        "matched_gt_ids_50": matched_gt_50,
        "matched_pred_ids_50": threshold_results[0.5]["matched_pred_ids"],
        "duplicate_pred_ids_50": threshold_results[0.5]["duplicate_pred_ids"],
    }


def _max_iou_against_gt(prediction: PredictionBox, gt_boxes: Sequence[GroundTruthBox]) -> float:
    return max((_compute_iou(prediction.bbox_xyxy, gt.bbox_xyxy) for gt in gt_boxes), default=0.0)


def _classify_gt_failure(
    gt: GroundTruthBox,
    image_predictions: Sequence[PredictionBox],
    prompt: PromptEntry,
    localization_min_iou: float,
    match_iou_threshold: float,
) -> GTFailureAssignment:
    best_prediction: Optional[PredictionBox] = None
    best_iou = 0.0
    best_same_class_iou = 0.0

    for pred in image_predictions:
        iou = _compute_iou(pred.bbox_xyxy, gt.bbox_xyxy)
        if pred.class_name == gt.category_name and iou > best_same_class_iou:
            best_same_class_iou = iou
        if best_prediction is None or (iou, pred.confidence) > (best_iou, best_prediction.confidence):
            best_prediction = pred
            best_iou = iou

    predicted_label = best_prediction.class_name if best_prediction else None
    if best_prediction is not None and best_iou >= match_iou_threshold:
        if predicted_label == gt.category_name:
            category = "correct"
        elif predicted_label in prompt.confusables:
            category = "semantic_confusion"
        else:
            category = "wrong_class"
    elif best_same_class_iou >= localization_min_iou:
        category = "localization_error"
    else:
        category = "miss"

    return GTFailureAssignment(
        category=category,
        gt_label=gt.category_name,
        predicted_label=predicted_label,
        best_iou=best_iou,
        best_same_class_iou=best_same_class_iou,
    )


def _presence_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    total = len(rows)
    return {
        "total_images": total,
        "target_accuracy": _safe_div(sum(row["image_level_target_detected"] for row in rows), total),
        "confusion_rate": _safe_div(sum(row["image_level_confusion_detected"] for row in rows), total),
        "hallucination_rate": _safe_div(sum(row["image_level_hallucination_detected"] for row in rows), total),
    }


def _compute_failure_rows(
    prompts: Dict[str, PromptEntry],
    gt_by_image: Dict[str, List[GroundTruthBox]],
    predictions_by_image: Dict[str, List[PredictionBox]],
    image_ids: Sequence[str],
    matched_gt_ids_50: Set[int],
    matched_pred_ids_50: Set[str],
    duplicate_pred_ids_50: Set[str],
    localization_min_iou: float,
    match_iou_threshold: float,
    hallucination_max_iou: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    confusion_counter: Counter[Tuple[str, str, str]] = Counter()
    confusion_iou_sum: Dict[Tuple[str, str, str], float] = defaultdict(float)
    rows: List[Dict[str, Any]] = []

    for image_id in sorted(image_ids):
        prompt = prompts[image_id]
        gt_boxes = gt_by_image.get(image_id, [])
        image_predictions = predictions_by_image.get(image_id, [])

        failure_counts: Counter[str] = Counter()
        for gt in gt_boxes:
            assignment = _classify_gt_failure(
                gt=gt,
                image_predictions=image_predictions,
                prompt=prompt,
                localization_min_iou=localization_min_iou,
                match_iou_threshold=match_iou_threshold,
            )
            failure_counts[assignment.category] += 1
            if assignment.category in {"semantic_confusion", "wrong_class"} and assignment.predicted_label:
                key = (assignment.category, assignment.gt_label, assignment.predicted_label)
                confusion_counter[key] += 1
                confusion_iou_sum[key] += assignment.best_iou

        hallucination_counts: Counter[str] = Counter()
        for pred in image_predictions:
            if pred.pred_id in matched_pred_ids_50:
                continue
            max_iou = _max_iou_against_gt(prediction=pred, gt_boxes=gt_boxes)
            if max_iou < hallucination_max_iou:
                hallucination_counts[prompt.label_group(pred.class_name)] += 1

        predicted_labels = {pred.class_name for pred in image_predictions}
        matched_gt_count = sum(1 for gt in gt_boxes if gt.annotation_id in matched_gt_ids_50)
        duplicate_count = sum(
            1 for pred in image_predictions if pred.pred_id in duplicate_pred_ids_50
        )

        rows.append(
            {
                "image_id": image_id,
                "scene_type": prompt.scene_type,
                "gt_count": len(gt_boxes),
                "tp50": matched_gt_count,
                "fn50": len(gt_boxes) - matched_gt_count,
                "correct_count": failure_counts["correct"],
                "miss_count": failure_counts["miss"],
                "semantic_confusion_count": failure_counts["semantic_confusion"],
                "wrong_class_count": failure_counts["wrong_class"],
                "localization_error_count": failure_counts["localization_error"],
                "duplicate_count": duplicate_count,
                "hallucination_total_count": sum(hallucination_counts.values()),
                "hallucination_target_count": hallucination_counts["target"],
                "hallucination_confusable_count": hallucination_counts["confusable"],
                "hallucination_distractor_count": hallucination_counts["distractor"],
                "hallucination_unknown_count": hallucination_counts["unknown"],
                "image_level_target_detected": int(bool(predicted_labels & prompt.targets)),
                "image_level_confusion_detected": int(bool(predicted_labels & prompt.confusables)),
                "image_level_hallucination_detected": int(bool(predicted_labels & prompt.distractors)),
            }
        )

    confusion_rows = [
        {
            "failure_type": failure_type,
            "gt_label": gt_label,
            "predicted_label": predicted_label,
            "count": count,
            "mean_iou": round(confusion_iou_sum[(failure_type, gt_label, predicted_label)] / count, 4),
        }
        for (failure_type, gt_label, predicted_label), count in sorted(
            confusion_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    return rows, confusion_rows


def _validate_prompt_coverage(
    prompts: Dict[str, PromptEntry],
    gt_by_image: Dict[str, List[GroundTruthBox]],
    image_ids: Sequence[str],
) -> Dict[str, Any]:
    missing_targets: List[Dict[str, Any]] = []
    for image_id in image_ids:
        prompt = prompts[image_id]
        gt_labels = {gt.category_name for gt in gt_by_image.get(image_id, [])}
        missing = sorted(gt_labels - set(prompt.targets))
        if missing:
            missing_targets.append({"image_id": image_id, "missing_targets": missing})

    return {
        "targets_cover_all_gt": not missing_targets,
        "missing_targets": missing_targets,
    }


def _summarize_group(
    group_rows: Sequence[Dict[str, Any]],
    detection_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    num_images = len(group_rows)
    num_gt = sum(row["gt_count"] for row in group_rows)
    correct_count = sum(row["correct_count"] for row in group_rows)
    miss_count = sum(row["miss_count"] for row in group_rows)
    semantic_confusion_count = sum(row["semantic_confusion_count"] for row in group_rows)
    wrong_class_count = sum(row["wrong_class_count"] for row in group_rows)
    localization_error_count = sum(row["localization_error_count"] for row in group_rows)
    duplicate_count = sum(row["duplicate_count"] for row in group_rows)

    summary = {
        "num_images": num_images,
        "num_gt": detection_metrics["num_gt"],
        "num_classes": detection_metrics["num_classes"],
        "num_target_predictions": detection_metrics["num_target_predictions"],
        "ap50": round(detection_metrics["ap50"], 4),
        "ap75": round(detection_metrics["ap75"], 4),
        "map50_95": round(detection_metrics["map50_95"], 4),
        "recall50": round(detection_metrics["recall50"], 4),
        "ar100": round(detection_metrics["ar100"], 4),
        "recall_small": round(detection_metrics["recall_small"], 4) if detection_metrics["recall_small"] is not None else None,
        "recall_medium": round(detection_metrics["recall_medium"], 4) if detection_metrics["recall_medium"] is not None else None,
        "recall_large": round(detection_metrics["recall_large"], 4) if detection_metrics["recall_large"] is not None else None,
        "tp50": detection_metrics["tp50"],
        "fn50": detection_metrics["fn50"],
        "correct_count": correct_count,
        "miss_count": miss_count,
        "semantic_confusion_count": semantic_confusion_count,
        "wrong_class_count": wrong_class_count,
        "localization_error_count": localization_error_count,
        "duplicate_count": duplicate_count,
        "miss_rate": round(_safe_div(miss_count, num_gt), 4),
        "semantic_confusion_rate": round(_safe_div(semantic_confusion_count, num_gt), 4),
        "wrong_class_rate": round(_safe_div(wrong_class_count, num_gt), 4),
        "localization_error_rate": round(_safe_div(localization_error_count, num_gt), 4),
        "duplicate_rate": round(_safe_div(duplicate_count, num_gt), 4),
        "hallucination_total_fp_per_image": round(
            _safe_div(sum(row["hallucination_total_count"] for row in group_rows), num_images),
            4,
        ),
        "hallucination_target_fp_per_image": round(
            _safe_div(sum(row["hallucination_target_count"] for row in group_rows), num_images),
            4,
        ),
        "hallucination_confusable_fp_per_image": round(
            _safe_div(sum(row["hallucination_confusable_count"] for row in group_rows), num_images),
            4,
        ),
        "hallucination_distractor_fp_per_image": round(
            _safe_div(sum(row["hallucination_distractor_count"] for row in group_rows), num_images),
            4,
        ),
        "hallucination_unknown_fp_per_image": round(
            _safe_div(sum(row["hallucination_unknown_count"] for row in group_rows), num_images),
            4,
        ),
        "image_level_presence": _presence_summary(group_rows),
    }
    if num_gt == 0 and detection_metrics["num_gt"] != 0:
        summary["num_gt"] = num_gt
    return summary


def run_evaluation(
    predictions_path: str,
    gt_json_path: str,
    prompts_csv_path: str,
    max_dets: int = 100,
    localization_min_iou: float = 0.1,
    match_iou_threshold: float = 0.5,
    hallucination_max_iou: float = 0.1,
) -> EvaluationArtifacts:
    prompts = load_prompts(prompts_csv_path)
    image_info_by_name, gt_by_image = load_ground_truth(gt_json_path)
    prediction_metadata, predictions_by_image = load_predictions(predictions_path)

    image_ids = sorted(prompts)
    missing_images = [image_id for image_id in image_ids if image_id not in image_info_by_name]
    if missing_images:
        missing_preview = ", ".join(missing_images[:5])
        raise ValueError(f"Prompt image ids not found in GT JSON: {missing_preview}")

    prompt_coverage = _validate_prompt_coverage(
        prompts=prompts,
        gt_by_image=gt_by_image,
        image_ids=image_ids,
    )

    overall_detection = _compute_detection_metrics(
        prompts=prompts,
        gt_by_image=gt_by_image,
        predictions_by_image=predictions_by_image,
        image_ids=image_ids,
        iou_thresholds=IOU_THRESHOLDS,
        max_dets=max_dets,
    )
    per_image_rows, confusion_rows = _compute_failure_rows(
        prompts=prompts,
        gt_by_image=gt_by_image,
        predictions_by_image=predictions_by_image,
        image_ids=image_ids,
        matched_gt_ids_50=overall_detection["matched_gt_ids_50"],
        matched_pred_ids_50=overall_detection["matched_pred_ids_50"],
        duplicate_pred_ids_50=overall_detection["duplicate_pred_ids_50"],
        localization_min_iou=localization_min_iou,
        match_iou_threshold=match_iou_threshold,
        hallucination_max_iou=hallucination_max_iou,
    )

    groups: Dict[str, List[str]] = {"overall": image_ids}
    for image_id, prompt in prompts.items():
        groups.setdefault(prompt.scene_type, []).append(image_id)

    metrics_by_group: Dict[str, Dict[str, Any]] = {}
    presence_by_group: Dict[str, Dict[str, float]] = {}
    for group_name, group_image_ids in groups.items():
        detection_metrics = (
            overall_detection
            if group_name == "overall"
            else _compute_detection_metrics(
                prompts=prompts,
                gt_by_image=gt_by_image,
                predictions_by_image=predictions_by_image,
                image_ids=sorted(group_image_ids),
                iou_thresholds=IOU_THRESHOLDS,
                max_dets=max_dets,
            )
        )
        group_rows = [
            row for row in per_image_rows
            if group_name == "overall" or row["scene_type"] == group_name
        ]
        group_summary = _summarize_group(group_rows=group_rows, detection_metrics=detection_metrics)
        metrics_by_group[group_name] = group_summary
        presence_by_group[group_name] = group_summary["image_level_presence"]

    zero_gt_images = [
        image_id
        for image_id in image_ids
        if not gt_by_image.get(image_id)
    ]

    summary = {
        "metric_backend": "native_coco_style",
        "config": {
            "predictions": str(Path(predictions_path).resolve()),
            "gt_json": str(Path(gt_json_path).resolve()),
            "prompts_csv": str(Path(prompts_csv_path).resolve()),
            "iou_thresholds": list(IOU_THRESHOLDS),
            "max_dets": max_dets,
            "localization_min_iou": localization_min_iou,
            "match_iou_threshold": match_iou_threshold,
            "hallucination_max_iou": hallucination_max_iou,
        },
        "prediction_metadata": prediction_metadata,
        "dataset": {
            "num_images": len(image_ids),
            "num_gt_annotations": sum(len(gt_by_image.get(image_id, [])) for image_id in image_ids),
            "scene_counts": {
                scene: len(group_image_ids)
                for scene, group_image_ids in groups.items()
                if scene != "overall"
            },
            "zero_gt_images": zero_gt_images,
            **prompt_coverage,
        },
        "metrics": metrics_by_group,
        "appendix": {
            "image_level_presence": presence_by_group,
        },
    }

    return EvaluationArtifacts(
        summary=summary,
        per_image_rows=per_image_rows,
        confusion_rows=confusion_rows,
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(artifacts: EvaluationArtifacts, outdir: str) -> Dict[str, str]:
    outdir_path = Path(outdir).expanduser().resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    summary_path = outdir_path / "metrics_summary.json"
    per_image_path = outdir_path / "per_image_failures.csv"
    confusion_path = outdir_path / "confusion_pairs.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.summary, handle, ensure_ascii=False, indent=2)

    _write_csv(per_image_path, artifacts.per_image_rows)
    _write_csv(confusion_path, artifacts.confusion_rows)

    return {
        "metrics_summary": str(summary_path),
        "per_image_failures": str(per_image_path),
        "confusion_pairs": str(confusion_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run box-level evaluation and failure-case analysis for YOLO-World outputs."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/predictions.json",
        help="Path to predictions JSON exported by yoloworld-usage.py.",
    )
    parser.add_argument(
        "--gt-json",
        type=str,
        default="images/instances_val2017_select.json",
        help="Path to the selected COCO-style GT annotations JSON.",
    )
    parser.add_argument(
        "--prompts-csv",
        "--csv",
        dest="prompts_csv",
        type=str,
        default="prompts/prompts.csv",
        help="Path to prompts CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/eval_box",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=100,
        help="Max target detections per image used in standard metrics.",
    )
    parser.add_argument(
        "--localization-min-iou",
        type=float,
        default=0.1,
        help="Minimum IoU for labeling a same-class miss as localization_error.",
    )
    parser.add_argument(
        "--hallucination-max-iou",
        type=float,
        default=0.1,
        help="Predictions with max IoU below this threshold count as hallucinations.",
    )
    args = parser.parse_args()

    artifacts = run_evaluation(
        predictions_path=args.predictions,
        gt_json_path=args.gt_json,
        prompts_csv_path=args.prompts_csv,
        max_dets=int(args.max_dets),
        localization_min_iou=float(args.localization_min_iou),
        hallucination_max_iou=float(args.hallucination_max_iou),
    )
    output_paths = write_outputs(artifacts=artifacts, outdir=args.outdir)

    overall = artifacts.summary["metrics"]["overall"]
    print(f"Images:                 {artifacts.summary['dataset']['num_images']}")
    print(f"GT annotations:         {artifacts.summary['dataset']['num_gt_annotations']}")
    print(f"AP50:                   {overall['ap50']:.4f}")
    print(f"AP75:                   {overall['ap75']:.4f}")
    print(f"mAP@[0.50:0.95]:        {overall['map50_95']:.4f}")
    print(f"Recall@0.5:             {overall['recall50']:.4f}")
    print(f"AR@100:                 {overall['ar100']:.4f}")
    print(f"Miss rate:              {overall['miss_rate']:.4f}")
    print(f"Semantic confusion:     {overall['semantic_confusion_rate']:.4f}")
    print(f"Wrong-class rate:       {overall['wrong_class_rate']:.4f}")
    print(f"Localization error:     {overall['localization_error_rate']:.4f}")
    print(f"Duplicate rate:         {overall['duplicate_rate']:.4f}")
    print(
        "Hallucination FP/image: "
        f"{overall['hallucination_total_fp_per_image']:.4f}"
    )
    print()
    print(f"Metrics JSON:           {output_paths['metrics_summary']}")
    print(f"Per-image failures CSV: {output_paths['per_image_failures']}")
    print(f"Confusion pairs CSV:    {output_paths['confusion_pairs']}")


if __name__ == "__main__":
    main()
