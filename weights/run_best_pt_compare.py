#!/usr/bin/env python3
"""Chạy best.pt FP32 trên thư mục ảnh và xuất dữ liệu để so sánh với XModel.

Kết quả:
- annotated/: ảnh đã vẽ bounding box
- predictions.csv: từng detection
- per_image.csv: thống kê từng ảnh
- summary.json: thống kê tổng hợp
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Đường dẫn best.pt")
    parser.add_argument("--images", required=True, help="Thư mục 20 ảnh test")
    parser.add_argument("--output", default="fp32_results")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="cpu hoặc 0")
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def collect_images(folder):
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Không tìm thấy thư mục ảnh: {}".format(root))

    images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError("Không có ảnh hợp lệ trong {}".format(root))
    return images


def main():
    args = parse_args()

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError("Không tìm thấy weights: {}".format(weights))

    image_paths = collect_images(args.images)
    output_dir = Path(args.output).expanduser().resolve()
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    print("Weights:", weights)
    print("Images :", len(image_paths))
    print("Output :", output_dir)

    model = YOLO(str(weights))
    print("Classes:", model.names)

    detection_rows = []
    image_rows = []
    total_start = time.perf_counter()

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        rect=False,          # ép letterbox vuông 640x640 giống XModel
        half=False,          # giữ FP32
        augment=False,
        agnostic_nms=False,  # NMS riêng theo từng class
        verbose=False,
        stream=True,
    )

    for index, (image_path, result) in enumerate(zip(image_paths, results), start=1):
        boxes = result.boxes
        num_detections = 0 if boxes is None else len(boxes)

        speed = result.speed or {}
        preprocess_ms = float(speed.get("preprocess", 0.0))
        inference_ms = float(speed.get("inference", 0.0))
        postprocess_ms = float(speed.get("postprocess", 0.0))
        total_ms = preprocess_ms + inference_ms + postprocess_ms

        per_class = {}
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for det_index, (box, score, class_id) in enumerate(
                zip(xyxy, confs, class_ids)
            ):
                class_name = model.names[int(class_id)]
                per_class[class_name] = per_class.get(class_name, 0) + 1

                x1, y1, x2, y2 = [float(value) for value in box]
                detection_rows.append({
                    "image": image_path.name,
                    "detection_id": det_index,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                })

        annotated = result.plot()
        cv2.imwrite(str(annotated_dir / image_path.name), annotated)

        image_rows.append({
            "image": image_path.name,
            "detections": num_detections,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
            "class_counts": json.dumps(per_class, ensure_ascii=False),
        })

        print(
            "[{}/{}] {}: {} detections, inference={:.3f} ms".format(
                index,
                len(image_paths),
                image_path.name,
                num_detections,
                inference_ms,
            )
        )

    elapsed = time.perf_counter() - total_start

    predictions_csv = output_dir / "predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "image", "detection_id", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "width", "height"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detection_rows)

    per_image_csv = output_dir / "per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "image", "detections", "preprocess_ms", "inference_ms",
            "postprocess_ms", "total_ms", "class_counts"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)

    inference_values = [row["inference_ms"] for row in image_rows]
    total_values = [row["total_ms"] for row in image_rows]

    class_totals = {}
    for row in detection_rows:
        name = row["class_name"]
        class_totals[name] = class_totals.get(name, 0) + 1

    summary = {
        "weights": str(weights),
        "images_folder": str(Path(args.images).expanduser().resolve()),
        "number_of_images": len(image_paths),
        "number_of_detections": len(detection_rows),
        "class_totals": class_totals,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "mean_inference_ms": float(np.mean(inference_values)),
        "median_inference_ms": float(np.median(inference_values)),
        "min_inference_ms": float(np.min(inference_values)),
        "max_inference_ms": float(np.max(inference_values)),
        "mean_pipeline_ms": float(np.mean(total_values)),
        "approximate_inference_fps": (
            1000.0 / float(np.mean(inference_values))
            if np.mean(inference_values) > 0 else None
        ),
        "wall_time_seconds": elapsed,
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDONE")
    print("Annotated images :", annotated_dir)
    print("Predictions CSV  :", predictions_csv)
    print("Per-image CSV    :", per_image_csv)
    print("Summary JSON     :", summary_json)
    print("Mean inference   : {:.3f} ms".format(summary["mean_inference_ms"]))
    print("Approx. FPS      : {:.2f}".format(summary["approximate_inference_fps"]))


if __name__ == "__main__":
    main()
