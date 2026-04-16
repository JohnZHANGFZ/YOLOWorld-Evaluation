from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import evaluate


class EvaluateSyntheticFixtureTests(unittest.TestCase):
    def _write_fixture_files(self, tmpdir: Path) -> tuple[Path, Path, Path]:
        prompts_path = tmpdir / "prompts.csv"
        gt_path = tmpdir / "gt.json"
        predictions_path = tmpdir / "predictions.json"

        prompts_rows = [
            {
                "image_id": "img_correct.jpg",
                "scene_type": "natural",
                "targets": "cat",
                "confusables": "dog",
                "distractors": "car",
            },
            {
                "image_id": "img_confusion.jpg",
                "scene_type": "ambiguous",
                "targets": "cup",
                "confusables": "mug",
                "distractors": "plate",
            },
            {
                "image_id": "img_duplicate.jpg",
                "scene_type": "natural",
                "targets": "dog",
                "confusables": "wolf",
                "distractors": "ball",
            },
            {
                "image_id": "img_localization.jpg",
                "scene_type": "unusual",
                "targets": "bus",
                "confusables": "truck",
                "distractors": "pole",
            },
            {
                "image_id": "img_hallucination.jpg",
                "scene_type": "unusual",
                "targets": "chair",
                "confusables": "couch",
                "distractors": "lamp",
            },
            {
                "image_id": "img_wrong_class.jpg",
                "scene_type": "ambiguous",
                "targets": "chair",
                "confusables": "couch",
                "distractors": "lamp",
            },
        ]
        with prompts_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "scene_type", "targets", "confusables", "distractors"],
            )
            writer.writeheader()
            writer.writerows(prompts_rows)

        gt_payload = {
            "images": [
                {"id": 1, "file_name": "img_correct.jpg", "width": 200, "height": 200},
                {"id": 2, "file_name": "img_confusion.jpg", "width": 200, "height": 200},
                {"id": 3, "file_name": "img_duplicate.jpg", "width": 200, "height": 200},
                {"id": 4, "file_name": "img_localization.jpg", "width": 200, "height": 200},
                {"id": 5, "file_name": "img_hallucination.jpg", "width": 200, "height": 200},
                {"id": 6, "file_name": "img_wrong_class.jpg", "width": 200, "height": 200},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "cup"},
                {"id": 3, "name": "dog"},
                {"id": 4, "name": "bus"},
                {"id": 5, "name": "chair"},
            ],
            "annotations": [
                {"id": 101, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "area": 10000},
                {"id": 102, "image_id": 2, "category_id": 2, "bbox": [0, 0, 60, 60], "area": 3600},
                {"id": 103, "image_id": 3, "category_id": 3, "bbox": [0, 0, 80, 80], "area": 6400},
                {"id": 104, "image_id": 4, "category_id": 4, "bbox": [0, 0, 100, 100], "area": 10000},
                {"id": 105, "image_id": 6, "category_id": 5, "bbox": [0, 0, 50, 50], "area": 2500},
            ],
        }
        gt_path.write_text(json.dumps(gt_payload), encoding="utf-8")

        predictions_payload = {
            "predictions": [
                {
                    "image": str(tmpdir / "img_correct.jpg"),
                    "detections": [
                        {"bbox_xyxy": [0, 0, 100, 100], "confidence": 0.99, "class_name": "cat"},
                    ],
                },
                {
                    "image": str(tmpdir / "img_confusion.jpg"),
                    "detections": [
                        {"bbox_xyxy": [0, 0, 60, 60], "confidence": 0.95, "class_name": "mug"},
                        {"bbox_xyxy": [30, 30, 90, 90], "confidence": 0.70, "class_name": "cup"},
                    ],
                },
                {
                    "image": str(tmpdir / "img_duplicate.jpg"),
                    "detections": [
                        {"bbox_xyxy": [0, 0, 80, 80], "confidence": 0.95, "class_name": "dog"},
                        {"bbox_xyxy": [0, 0, 80, 80], "confidence": 0.60, "class_name": "dog"},
                    ],
                },
                {
                    "image": str(tmpdir / "img_localization.jpg"),
                    "detections": [
                        {"bbox_xyxy": [50, 50, 150, 150], "confidence": 0.88, "class_name": "bus"},
                    ],
                },
                {
                    "image": str(tmpdir / "img_hallucination.jpg"),
                    "detections": [
                        {"bbox_xyxy": [10, 10, 60, 60], "confidence": 0.92, "class_name": "couch"},
                        {"bbox_xyxy": [70, 70, 120, 120], "confidence": 0.70, "class_name": "lamp"},
                    ],
                },
                {
                    "image": str(tmpdir / "img_wrong_class.jpg"),
                    "detections": [
                        {"bbox_xyxy": [0, 0, 50, 50], "confidence": 0.91, "class_name": "lamp"},
                    ],
                },
            ]
        }
        predictions_path.write_text(json.dumps(predictions_payload), encoding="utf-8")
        return prompts_path, gt_path, predictions_path

    def test_loader_maps_category_names_and_size_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _, gt_path, _ = self._write_fixture_files(tmpdir)

            images_by_name, gt_by_image = evaluate.load_ground_truth(str(gt_path))

            self.assertIn("img_correct.jpg", images_by_name)
            cat_gt = gt_by_image["img_correct.jpg"][0]
            self.assertEqual(cat_gt.category_name, "cat")
            self.assertEqual(cat_gt.size_bucket, "large")

    def test_synthetic_fixture_covers_failure_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            prompts_path, gt_path, predictions_path = self._write_fixture_files(tmpdir)

            artifacts = evaluate.run_evaluation(
                predictions_path=str(predictions_path),
                gt_json_path=str(gt_path),
                prompts_csv_path=str(prompts_path),
            )

            summary = artifacts.summary
            overall = summary["metrics"]["overall"]

            self.assertEqual(summary["dataset"]["num_images"], 6)
            self.assertEqual(summary["dataset"]["scene_counts"], {"natural": 2, "ambiguous": 2, "unusual": 2})
            self.assertTrue(summary["dataset"]["targets_cover_all_gt"])
            self.assertEqual(summary["dataset"]["zero_gt_images"], ["img_hallucination.jpg"])

            self.assertEqual(overall["tp50"], 2)
            self.assertEqual(overall["fn50"], 3)
            self.assertEqual(overall["correct_count"], 2)
            self.assertEqual(overall["semantic_confusion_count"], 1)
            self.assertEqual(overall["localization_error_count"], 1)
            self.assertEqual(overall["wrong_class_count"], 1)
            self.assertEqual(overall["duplicate_count"], 1)
            self.assertEqual(overall["miss_count"], 0)
            self.assertAlmostEqual(overall["recall50"], 0.4, places=4)
            self.assertAlmostEqual(overall["duplicate_rate"], 0.2, places=4)

            per_image = {row["image_id"]: row for row in artifacts.per_image_rows}
            self.assertEqual(per_image["img_duplicate.jpg"]["duplicate_count"], 1)
            self.assertEqual(per_image["img_confusion.jpg"]["semantic_confusion_count"], 1)
            self.assertEqual(per_image["img_localization.jpg"]["localization_error_count"], 1)
            self.assertEqual(per_image["img_wrong_class.jpg"]["wrong_class_count"], 1)
            self.assertEqual(per_image["img_hallucination.jpg"]["hallucination_confusable_count"], 1)
            self.assertEqual(per_image["img_hallucination.jpg"]["hallucination_distractor_count"], 1)

            confusion_pairs = {
                (row["failure_type"], row["gt_label"], row["predicted_label"]): row["count"]
                for row in artifacts.confusion_rows
            }
            self.assertEqual(confusion_pairs[("semantic_confusion", "cup", "mug")], 1)
            self.assertEqual(confusion_pairs[("wrong_class", "chair", "lamp")], 1)


class EvaluateRealSubsetSmokeTests(unittest.TestCase):
    def test_real_subset_outputs_grouped_metrics(self) -> None:
        root = Path(__file__).resolve().parents[1]
        artifacts = evaluate.run_evaluation(
            predictions_path=str(root / "outputs/predictions.json"),
            gt_json_path=str(root / "images/instances_val2017_select.json"),
            prompts_csv_path=str(root / "prompts/prompts.csv"),
        )

        summary = artifacts.summary
        self.assertEqual(summary["dataset"]["scene_counts"], {"ambiguous": 34, "natural": 34, "unusual": 35})
        self.assertTrue(summary["dataset"]["targets_cover_all_gt"])
        self.assertEqual(len(artifacts.per_image_rows), 103)
        for group_name in ["overall", "natural", "ambiguous", "unusual"]:
            self.assertIn(group_name, summary["metrics"])
            self.assertIn("ap50", summary["metrics"][group_name])
            self.assertIn("recall50", summary["metrics"][group_name])


if __name__ == "__main__":
    unittest.main()
