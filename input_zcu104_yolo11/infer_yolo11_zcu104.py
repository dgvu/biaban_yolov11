#!/usr/bin/env python3
import argparse
import json
import os
import time

import cv2
import numpy as np
import vart
import xir

CLASS_NAMES = ["bullet", "target"]
REG_MAX = 16
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def dpu_subgraphs(graph):
    root = graph.get_root_subgraph()
    return [
        s for s in root.toposort_child_subgraph()
        if s.has_attr("device") and str(s.get_attr("device")).upper() == "DPU"
    ]


def fix_point(tensor):
    return int(tensor.get_attr("fix_point"))


def np_dtype(tensor):
    name = str(tensor.dtype)
    if name.startswith("x"):
        name = name[1:]
    table = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "float32": np.float32}
    if name not in table:
        raise RuntimeError("Unsupported dtype: {} ({})".format(tensor.name, tensor.dtype))
    return table[name]


def letterbox(image, target_h, target_w):
    h, w = image.shape[:2]
    ratio = min(float(target_w) / w, float(target_h) / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, {"ratio": ratio, "left": left, "top": top, "src_w": w, "src_h": h}


def preprocess(path, tensor):
    dims = tuple(tensor.dims)
    if dims[-1] == 3:
        layout, h, w = "NHWC", dims[1], dims[2]
    elif dims[1] == 3:
        layout, h, w = "NCHW", dims[2], dims[3]
    else:
        raise RuntimeError("Cannot determine input layout: {}".format(dims))

    original = cv2.imread(path, cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(path)

    image, meta = letterbox(original, h, w)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":
        image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)

    dtype = np_dtype(tensor)
    quantized = np.rint(image * float(2 ** fix_point(tensor)))
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        quantized = np.clip(quantized, limits.min, limits.max)
    return original, np.ascontiguousarray(quantized.astype(dtype)), meta


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def to_hwc(array):
    array = array[0]
    if array.shape[-1] in (2, 64):
        return array
    if array.shape[0] in (2, 64):
        return array.transpose(1, 2, 0)
    raise RuntimeError("Unknown output layout: {}".format(array.shape))


def classify_output(array, input_size):
    hwc = to_hwc(array)
    h, w, c = hwc.shape
    if h != w or input_size % h != 0:
        raise RuntimeError("Bad output shape: {}".format(hwc.shape))
    stride = input_size // h
    if c == 64:
        kind = "box"
    elif c == len(CLASS_NAMES):
        kind = "cls"
    else:
        raise RuntimeError("Bad channel count: {}".format(hwc.shape))
    return kind, stride, hwc


def decode_head(box_logits, cls_logits, stride, conf):
    h, w, c = box_logits.shape
    box_logits = box_logits.reshape(h, w, 4, REG_MAX)
    probs = softmax(box_logits)
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, 1, REG_MAX)
    dist = np.sum(probs * bins, axis=-1)

    cls_scores = sigmoid(cls_logits)
    class_ids = np.argmax(cls_scores, axis=-1)
    scores = np.max(cls_scores, axis=-1)
    ys, xs = np.nonzero(scores >= conf)
    if len(xs) == 0:
        return (np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))

    d = dist[ys, xs]
    ax = xs.astype(np.float32) + 0.5
    ay = ys.astype(np.float32) + 0.5
    boxes = np.stack([
        (ax - d[:, 0]) * stride,
        (ay - d[:, 1]) * stride,
        (ax + d[:, 2]) * stride,
        (ay + d[:, 3]) * stride,
    ], axis=1)
    return boxes, scores[ys, xs].astype(np.float32), class_ids[ys, xs].astype(np.int32)


def class_aware_nms(boxes, scores, classes, conf, iou, max_det):
    kept = []
    for class_id in np.unique(classes):
        idx = np.where(classes == class_id)[0]
        xywh = []
        sc = []
        for i in idx:
            x1, y1, x2, y2 = boxes[i]
            xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            sc.append(float(scores[i]))
        selected = cv2.dnn.NMSBoxes(xywh, sc, conf, iou)
        if selected is not None and len(selected):
            for j in np.asarray(selected).reshape(-1):
                kept.append(idx[int(j)])
    kept.sort(key=lambda i: float(scores[i]), reverse=True)
    return np.asarray(kept[:max_det], dtype=np.int32)


def restore_boxes(boxes, meta):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= meta["left"]
    boxes[:, [1, 3]] -= meta["top"]
    boxes /= meta["ratio"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta["src_w"] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta["src_h"] - 1)
    return boxes


def draw(image, boxes, scores, classes):
    result = image.copy()
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = np.rint(box).astype(np.int32)
        label = "{} {:.2f}".format(CLASS_NAMES[int(class_id)], float(score))
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return result


def resolve_image_path(images_dir, idx):
    """Look for an image named `idx` under images_dir, trying common extensions.
    Returns the path if found, otherwise None (caller decides how to report it)."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, "{}{}".format(idx, ext))
        if os.path.isfile(candidate):
            return candidate
    return None


def run_inference(runner, input_tensors, output_tensors, image_path, args):
    """Run the DPU on a single image. Returns a dict with the annotated image,
    detections, source image size, and separate DPU / post-process timings."""
    original, input_data, meta = preprocess(image_path, input_tensors[0])
    outputs = [np.empty(tuple(t.dims), dtype=np_dtype(t), order="C") for t in output_tensors]

    t0 = time.perf_counter()
    job = runner.execute_async([input_data], outputs)
    if runner.wait(job) != 0:
        raise RuntimeError("Inference failed for {}".format(image_path))
    t1 = time.perf_counter()

    dims = tuple(input_tensors[0].dims)
    input_size = dims[1] if dims[-1] == 3 else dims[2]
    heads = {}
    for tensor, raw in zip(output_tensors, outputs):
        fp32 = raw.astype(np.float32) * float(2 ** (-fix_point(tensor)))
        kind, stride, hwc = classify_output(fp32, input_size)
        heads[(kind, stride)] = hwc

    all_boxes, all_scores, all_classes = [], [], []
    for stride in (8, 16, 32):
        boxes, scores, classes = decode_head(
            heads[("box", stride)], heads[("cls", stride)], stride, args.conf
        )
        if len(boxes):
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    if all_boxes:
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        keep = class_aware_nms(boxes, scores, classes, args.conf, args.iou, args.max_det)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        boxes = restore_boxes(boxes, meta)
    else:
        boxes = np.empty((0, 4), np.float32)
        scores = np.empty((0,), np.float32)
        classes = np.empty((0,), np.int32)

    result_image = draw(original, boxes, scores, classes)

    detections = []
    for box, score, class_id in zip(boxes, scores, classes):
        detections.append({
            "class": CLASS_NAMES[int(class_id)],
            "class_id": int(class_id),
            "confidence": round(float(score), 4),
            "bbox_xyxy": [round(float(v), 2) for v in box.tolist()],
        })

    t2 = time.perf_counter()
    src_h, src_w = original.shape[:2]

    return {
        "result_image": result_image,
        "detections": detections,
        "width": int(src_w),
        "height": int(src_h),
        "dpu_ms": (t1 - t0) * 1000.0,
        "postprocess_ms": (t2 - t1) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    # Single-image mode (kept for backward compatibility).
    parser.add_argument("--image", help="Path to a single input image")
    parser.add_argument("--output", default="result.jpg", help="Output path for single-image mode")
    # Batch mode.
    parser.add_argument("--images-dir", help="Folder containing images named 1.jpg..N.jpg")
    parser.add_argument("--start-index", type=int, default=1, help="First image index (default: 1)")
    parser.add_argument("--end-index", type=int, default=20, help="Last image index (default: 20)")
    parser.add_argument("--results-dir", default="result_images", help="Folder to save annotated images")
    parser.add_argument("--json-output", default="detections.json", help="Path to the output JSON file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    if not args.image and not args.images_dir:
        parser.error("Provide either --image (single image) or --images-dir (batch folder)")

    graph = xir.Graph.deserialize(args.model)
    subs = dpu_subgraphs(graph)
    if len(subs) != 1:
        raise RuntimeError("Expected 1 DPU subgraph, got {}".format(len(subs)))

    runner = vart.Runner.create_runner(subs[0], "run")
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    if len(input_tensors) != 1 or len(output_tensors) != 6:
        raise RuntimeError("Expected 1 input and 6 outputs")

    if args.images_dir:
        indices = list(range(args.start_index, args.end_index + 1))
        resolved = {idx: resolve_image_path(args.images_dir, idx) for idx in indices}

        warm_idx = next((i for i in indices if resolved[i]), None)
        if warm_idx is None:
            raise RuntimeError(
                "No images found in {} for indices {}..{}".format(
                    args.images_dir, args.start_index, args.end_index
                )
            )
        _, warm_input, _ = preprocess(resolved[warm_idx], input_tensors[0])
        warm_outputs = [np.empty(tuple(t.dims), dtype=np_dtype(t), order="C") for t in output_tensors]
        job = runner.execute_async([warm_input], warm_outputs)
        if runner.wait(job) != 0:
            raise RuntimeError("Warm-up failed")

        os.makedirs(args.results_dir, exist_ok=True)

        images_json = []
        num_ok = 0
        num_failed = 0
        total_detections = 0
        batch_start = time.perf_counter()

        for idx in indices:
            path = resolved[idx]
            if path is None:
                msg = "no image found for index {} in {} (tried {})".format(
                    idx, args.images_dir, IMAGE_EXTS
                )
                images_json.append({"file": "{}.*".format(idx), "error": msg})
                num_failed += 1
                print("[{}] MISSING - {}".format(idx, msg))
                continue

            try:
                info = run_inference(runner, input_tensors, output_tensors, path, args)
            except Exception as e:
                images_json.append({"file": os.path.basename(path), "error": str(e)})
                num_failed += 1
                print("[{}] {} -> ERROR: {}".format(idx, os.path.basename(path), e))
                continue

            out_name = "{}.jpg".format(idx)
            out_path = os.path.join(args.results_dir, out_name)
            if not cv2.imwrite(out_path, info["result_image"]):
                raise RuntimeError("Cannot save {}".format(out_path))

            counts = {name: 0 for name in CLASS_NAMES}
            for d in info["detections"]:
                counts[d["class"]] += 1

            images_json.append({
                "file": os.path.basename(path),
                "result_file": os.path.join(args.results_dir, out_name),
                "width": info["width"],
                "height": info["height"],
                "num_detections": len(info["detections"]),
                "counts": counts,
                "dpu_ms": round(info["dpu_ms"], 3),
                "postprocess_ms": round(info["postprocess_ms"], 3),
                "detections": info["detections"],
            })

            num_ok += 1
            total_detections += len(info["detections"])
            print("[{}] {} -> {}: {} detection(s) {}, dpu={:.2f}ms post={:.2f}ms".format(
                idx, os.path.basename(path), out_name, len(info["detections"]),
                counts, info["dpu_ms"], info["postprocess_ms"]
            ))

        total_time_s = time.perf_counter() - batch_start

        output_json = {
            "xmodel": os.path.basename(args.model),
            "conf_thresh": args.conf,
            "iou": args.iou,
            "images": images_json,
            "summary": {
                "num_images": len(indices),
                "num_ok": num_ok,
                "num_failed": num_failed,
                "total_detections": total_detections,
                "total_time_s": round(total_time_s, 3),
            },
        }

        with open(args.json_output, "w") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        print("\nProcessed {} image(s): {} ok, {} failed.".format(len(indices), num_ok, num_failed))
        print("Total detections: {}".format(total_detections))
        print("Total time: {:.3f} s".format(total_time_s))
        print("Result images saved to:", os.path.abspath(args.results_dir))
        print("JSON saved to:", os.path.abspath(args.json_output))

    else:
        info = run_inference(runner, input_tensors, output_tensors, args.image, args)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        if not cv2.imwrite(args.output, info["result_image"]):
            raise RuntimeError("Cannot save {}".format(args.output))

        print("DPU time: {:.3f} ms".format(info["dpu_ms"]))
        print("Postprocess time: {:.3f} ms".format(info["postprocess_ms"]))
        print("Detections:", len(info["detections"]))
        for i, d in enumerate(info["detections"]):
            print("[{}] {} confidence={:.4f} bbox_xyxy={}".format(
                i, d["class"], d["confidence"], d["bbox_xyxy"]
            ))
        print("Saved:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
