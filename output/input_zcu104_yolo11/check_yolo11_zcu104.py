#!/usr/bin/env python3
import argparse
import os
import time

import cv2
import numpy as np
import vart
import xir


def get_dpu_subgraphs(graph):
    root = graph.get_root_subgraph()
    result = []
    for child in root.toposort_child_subgraph():
        if child.has_attr("device") and str(child.get_attr("device")).upper() == "DPU":
            result.append(child)
    return result


def fix_point(tensor):
    return int(tensor.get_attr("fix_point"))


def np_dtype(tensor):
    name = str(tensor.dtype)
    if name.startswith("x"):
        name = name[1:]
    mapping = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "float32": np.float32,
    }
    if name not in mapping:
        raise RuntimeError("Unsupported dtype: {} ({})".format(tensor.name, tensor.dtype))
    return mapping[name]


def letterbox(image, height, width):
    src_h, src_w = image.shape[:2]
    ratio = min(float(width) / src_w, float(height) / src_h)
    new_w = int(round(src_w * ratio))
    new_h = int(round(src_h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, {
        "ratio": ratio,
        "left": left,
        "top": top,
        "original_width": src_w,
        "original_height": src_h,
    }


def prepare_input(image_path, tensor):
    dims = tuple(tensor.dims)
    if len(dims) != 4 or dims[0] != 1:
        raise RuntimeError("Expected batch=1 input, got {}".format(dims))

    if dims[-1] == 3:
        layout = "NHWC"
        height, width = dims[1], dims[2]
    elif dims[1] == 3:
        layout = "NCHW"
        height, width = dims[2], dims[3]
    else:
        raise RuntimeError("Cannot determine input layout: {}".format(dims))

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    image, meta = letterbox(image, height, width)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    if layout == "NCHW":
        image = image.transpose(2, 0, 1)

    image = np.expand_dims(image, 0)
    dtype = np_dtype(tensor)
    scale = float(2 ** fix_point(tensor))
    quantized = np.rint(image * scale)

    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        quantized = np.clip(quantized, limits.min, limits.max)

    return np.ascontiguousarray(quantized.astype(dtype)), meta, layout


def role_from_shape(shape, input_size=640):
    if len(shape) != 4:
        return "unknown"

    if shape[-1] in (2, 64):
        height, width, channels = shape[1], shape[2], shape[3]
    elif shape[1] in (2, 64):
        channels, height, width = shape[1], shape[2], shape[3]
    else:
        return "unknown"

    if height <= 0 or input_size % height:
        return "unknown"

    stride = input_size // height
    return "{}_s{}".format("box" if channels == 64 else "cls", stride)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="raw_outputs")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    graph = xir.Graph.deserialize(args.model)
    dpu_subgraphs = get_dpu_subgraphs(graph)

    print("DPU subgraphs:", len(dpu_subgraphs))
    for index, subgraph in enumerate(dpu_subgraphs):
        print("  [{}] {}".format(index, subgraph.get_name()))

    if len(dpu_subgraphs) != 1:
        raise RuntimeError("Expected exactly 1 DPU subgraph")

    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    if len(input_tensors) != 1:
        raise RuntimeError("Expected 1 input tensor")
    if len(output_tensors) != 6:
        raise RuntimeError("Expected 6 output tensors, got {}".format(len(output_tensors)))

    print("\nINPUT")
    for tensor in input_tensors:
        print("  name={} shape={} dtype={} fix_point={}".format(
            tensor.name, tuple(tensor.dims), tensor.dtype, fix_point(tensor)
        ))

    print("\nOUTPUTS")
    for index, tensor in enumerate(output_tensors):
        print("  [{}] name={} shape={} dtype={} fix_point={} role={}".format(
            index,
            tensor.name,
            tuple(tensor.dims),
            tensor.dtype,
            fix_point(tensor),
            role_from_shape(tuple(tensor.dims)),
        ))

    input_data, meta, layout = prepare_input(args.image, input_tensors[0])
    outputs = [
        np.empty(tuple(tensor.dims), dtype=np_dtype(tensor), order="C")
        for tensor in output_tensors
    ]

    print("\nPREPROCESS")
    print("  layout:", layout)
    print("  input range:", int(input_data.min()), int(input_data.max()))
    print("  letterbox:", meta)

    job_id = runner.execute_async([input_data], outputs)
    status = runner.wait(job_id)
    if status != 0:
        raise RuntimeError("Warm-up failed: {}".format(status))

    repeat = max(1, args.repeat)
    start = time.perf_counter()
    for _ in range(repeat):
        job_id = runner.execute_async([input_data], outputs)
        status = runner.wait(job_id)
        if status != 0:
            raise RuntimeError("Inference failed: {}".format(status))
    elapsed = time.perf_counter() - start

    print("\nDPU RUN: PASS")
    print("  mean time: {:.3f} ms".format(elapsed * 1000.0 / repeat))
    print("  approximate FPS: {:.2f}".format(repeat / elapsed))

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nRAW OUTPUTS")
    for index, (tensor, raw) in enumerate(zip(output_tensors, outputs)):
        fp = fix_point(tensor)
        dequant = raw.astype(np.float32) * float(2 ** (-fp))
        role = role_from_shape(tuple(tensor.dims))
        if role == "unknown":
            role = "output_{:02d}".format(index)

        np.save(os.path.join(args.output_dir, role + "_int8.npy"), raw)
        np.save(os.path.join(args.output_dir, role + "_fp32.npy"), dequant)

        print("  {}: shape={} int=[{}, {}] fp=[{:.6f}, {:.6f}]".format(
            role,
            tuple(raw.shape),
            int(raw.min()),
            int(raw.max()),
            float(dequant.min()),
            float(dequant.max()),
        ))

    print("\nSaved:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
