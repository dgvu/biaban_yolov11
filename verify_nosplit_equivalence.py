#!/usr/bin/env python3
"""So sánh model C3k2 gốc và bản no-split sau khi chia trọng số."""

import argparse
import torch

from yolo11_standalone import load_standalone_checkpoint as load_original
from yolo11_standalone_nosplit import load_standalone_checkpoint as load_nosplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n_zcu104_state_dict.pth")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    torch.manual_seed(0)
    original, _ = load_original(args.weights, map_location="cpu")
    nosplit, _ = load_nosplit(args.weights, map_location="cpu")

    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        outputs_a = original(x)
        outputs_b = nosplit(x)

    worst = 0.0
    for index, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        error = float((a - b).abs().max().item())
        worst = max(worst, error)
        print("{} shape={} max_abs_error={:.9g}".format(
            index, tuple(a.shape), error
        ))

    print("Worst max_abs_error: {:.9g}".format(worst))
    if worst > 1e-5:
        raise RuntimeError("Hai model không tương đương trong ngưỡng 1e-5")
    print("EQUIVALENCE CHECK: PASS")


if __name__ == "__main__":
    main()
