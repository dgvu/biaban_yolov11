#!/usr/bin/env python3
"""Quantize YOLO11n standalone bằng Vitis AI PyTorch 3.0."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_nndct.apis import torch_quantizer

from yolo11_standalone_nosplit import load_standalone_checkpoint


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="yolo11n_zcu104_state_dict.pth",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Thư mục ảnh calib/test",
    )
    parser.add_argument(
        "--quant-mode",
        required=True,
        choices=("calib", "test"),
    )
    parser.add_argument(
        "--output-dir",
        default="quantize_result",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
    )
    parser.add_argument(
        "--deploy-check",
        action="store_true",
    )
    return parser.parse_args()


def collect_images(directory, limit):
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            "Không tìm thấy thư mục ảnh: {}".format(root)
        )

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not paths:
        raise RuntimeError(
            "Không có ảnh trong: {}".format(root)
        )

    if limit >= 0:
        paths = paths[:limit]

    return paths


def letterbox(image, size=640, pad_value=114):
    h, w = image.shape[:2]
    ratio = min(
        float(size) / float(h),
        float(size) / float(w),
    )

    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (size, size, 3),
        pad_value,
        dtype=np.uint8,
    )

    left = (size - new_w) // 2
    top = (size - new_h) // 2

    canvas[
        top:top + new_h,
        left:left + new_w,
    ] = resized

    return canvas


def preprocess(path, size):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError(
            "Không đọc được ảnh: {}".format(path)
        )

    image = letterbox(image, size)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)

    tensor = torch.from_numpy(image)
    tensor = tensor.float().div_(255.0)
    tensor = tensor.unsqueeze(0)
    return tensor


def print_shapes(outputs, title):
    print(title)
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(
            "Model không trả tuple/list."
        )
    if len(outputs) != 6:
        raise RuntimeError(
            "Mong đợi 6 outputs, nhận {}.".format(
                len(outputs)
            )
        )
    for index, output in enumerate(outputs):
        print(
            "  {}: shape={} dtype={}".format(
                index,
                tuple(output.shape),
                output.dtype,
            )
        )


def main():
    args = parse_args()

    if args.deploy and args.quant_mode != "test":
        raise ValueError(
            "--deploy chỉ dùng với --quant-mode test"
        )

    model, package = load_standalone_checkpoint(
        args.weights,
        map_location="cpu",
    )

    print("Checkpoint loaded strictly.")
    print("Classes:", package.get("names"))
    print(
        "State dict entries:",
        len(model.state_dict()),
    )

    dummy = torch.zeros(
        1,
        3,
        args.imgsz,
        args.imgsz,
    )

    with torch.no_grad():
        float_outputs = model(dummy)
    print_shapes(float_outputs, "FP32 raw outputs:")

    quantizer = torch_quantizer(
        args.quant_mode,
        model,
        (dummy,),
        output_dir=args.output_dir,
        device=torch.device("cpu"),
    )

    quant_model = quantizer.quant_model.eval()
    image_paths = collect_images(
        args.images,
        args.num_images,
    )

    if args.deploy:
        image_paths = image_paths[:1]

    print(
        "Running {} image(s) in {} mode".format(
            len(image_paths),
            args.quant_mode,
        )
    )

    with torch.no_grad():
        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            tensor = preprocess(
                image_path,
                args.imgsz,
            )
            outputs = quant_model(tensor)

            if index == 1:
                print_shapes(
                    outputs,
                    "Quantized raw outputs:",
                )

            if (
                index % 50 == 0
                or index == len(image_paths)
            ):
                print(
                    "[{}/{}] {}".format(
                        index,
                        len(image_paths),
                        image_path.name,
                    )
                )

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print(
            "Đã xuất quant_info.json vào:",
            args.output_dir,
        )

    if args.deploy:
        quantizer.export_xmodel(
            output_dir=args.output_dir,
            deploy_check=args.deploy_check,
        )
        print(
            "Đã yêu cầu xuất xmodel vào:",
            args.output_dir,
        )


if __name__ == "__main__":
    main()
