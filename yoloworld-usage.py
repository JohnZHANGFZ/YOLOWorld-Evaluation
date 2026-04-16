from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from ultralytics import YOLOWorld


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _build_image_index(images_dir: Path) -> Dict[str, Path]:
    """Build a mapping from filename -> absolute path for all images under images_dir."""
    index: Dict[str, Path] = {}
    for ext in sorted(IMG_EXTS):
        for p in images_dir.rglob(f"*{ext}"):
            if p.is_file():
                index[p.name] = p.resolve()
    return index


def _load_prompts(csv_path: str) -> List[Dict[str, Any]]:
    """Parse prompts CSV; combine targets / confusables / distractors as candidate labels."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()
            targets = [t.strip() for t in row["targets"].split(";") if t.strip()]
            confusables = [c.strip() for c in row["confusables"].split(";") if c.strip()]
            distractors = [d.strip() for d in row["distractors"].split(";") if d.strip()]
            labels = targets + confusables + distractors
            rows.append({
                "image_id": image_id,
                "labels": labels,
                "targets": targets,
                "confusables": confusables,
                "distractors": distractors,
            })
    return rows


def _result_to_dict(r: Any, image_path: str) -> Dict[str, Any]:
    # 将 Ultralytics 的 Results 对象转换为可序列化字典，
    # 便于统一写入 predictions.json 做后续评估/分析
    boxes = r.boxes
    names = getattr(r, "names", {}) or {}

    xyxy = boxes.xyxy.tolist() if getattr(boxes, "xyxy", None) is not None else []
    conf = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    cls = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []

    def _cls_name(cls_id: int) -> str:
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)

    dets = []
    for i in range(len(xyxy)):
        cls_id = int(cls[i]) if i < len(cls) else -1
        dets.append(
            {
                "bbox_xyxy": [float(v) for v in xyxy[i]],
                "confidence": float(conf[i]) if i < len(conf) else None,
                "class_id": cls_id,
                "class_name": _cls_name(cls_id),
            }
        )

    return {"image": image_path, "detections": dets}


def _pick_device(device_arg: Optional[str]) -> Optional[str]:
    # 设备选择策略：
    # 1) 用户显式传 --device 时优先使用该值
    if device_arg:
        return device_arg

    # Prefer MPS by default on Apple Silicon when available.
    # 2) 未显式指定时，在 Apple Silicon 上自动优先尝试 MPS
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"

    # 3) 其余情况返回 None，交给 Ultralytics 自动选择（通常是 CPU）
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run YOLO-World inference driven by a prompts CSV (per-image labels)."
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="yolov8s-worldv2.pt",
        help="Path to YOLO-World weights (default: yolov8s-worldv2.pt).",
    )
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to prompts CSV file (e.g. prompts/prompts.csv).",
    )
    ap.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Root directory containing images (searched recursively by filename).",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold (default: 0.001 for evaluation; use 0.25 for cleaner visualizations).",
    )
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string, e.g. "mps", "cpu", "0". Auto-selects MPS on Apple Silicon.',
    )
    ap.add_argument("--max", type=int, default=0, help="Max images to run (0 = no limit).")
    ap.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs).",
    )
    args = ap.parse_args()

    prompts = _load_prompts(args.csv)
    if not prompts:
        raise SystemExit("No entries found in the CSV file.")
    if args.max and args.max > 0:
        prompts = prompts[: args.max]

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    image_index = _build_image_index(images_dir)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    vis_dir = outdir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(args.weights)

    selected_device = _pick_device(args.device)
    predict_kwargs = dict(
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        verbose=False,
    )
    if selected_device is not None:
        predict_kwargs["device"] = selected_device

    preds: List[Dict[str, Any]] = []
    skipped = 0

    for i, entry in enumerate(prompts):
        image_id = entry["image_id"]
        img_path = image_index.get(image_id)
        if img_path is None:
            print(f"[SKIP] Image not found: {image_id}")
            skipped += 1
            continue

        # CLIP text encoder inside set_classes() cannot run on MPS;
        # temporarily move to CPU for text encoding, then restore device.
        model.model.cpu()
        model.set_classes(entry["labels"])
        if selected_device and selected_device != "cpu":
            model.model.to(selected_device)
        results = model.predict(str(img_path), **predict_kwargs)

        for r in results:
            r.save(filename=str(vis_dir / f"{img_path.stem}_labeled.jpg"))
            preds.append(_result_to_dict(r, str(img_path)))

        n_dets = preds[-1]["detections"].__len__() if preds else 0
        print(f"[{i + 1}/{len(prompts)}] {image_id}  labels={len(entry['labels'])}  dets={n_dets}")

    json_path = outdir / "predictions.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": str(Path(args.weights).resolve()),
                "classes": None,
                "csv": str(Path(args.csv).resolve()),
                "device": selected_device,
                "conf": float(args.conf),
                "imgsz": int(args.imgsz),
                "num_images": len(preds),
                "predictions": preds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDevice: {selected_device or 'auto'}")
    print(f"Processed: {len(preds)}, Skipped: {skipped}")
    print(f"Visualizations: {vis_dir}")
    print(f"Predictions JSON: {json_path}")


if __name__ == "__main__":
    # Avoid OpenMP over-subscription on some macOS setups
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()


"""
python yoloworld-usage.py \
    --csv prompts/prompts.csv \
    --images-dir images/ \
    --conf 0.25
"""
