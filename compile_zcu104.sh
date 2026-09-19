#!/usr/bin/env bash
set -euo pipefail

INPUT_XMODEL="${1:-quantize_result/DetectionModel_int.xmodel}"
OUTPUT_DIR="${2:-compiled_zcu104}"
NET_NAME="${3:-yolo11n_zcu104}"

if [[ ! -f "$INPUT_XMODEL" ]]; then
  echo "Không tìm thấy quantized xmodel: $INPUT_XMODEL" >&2
  exit 1
fi

ARCH="$(find /opt/vitis_ai/compiler/arch/DPUCZDX8G \
  -type f -path '*/ZCU104/arch.json' 2>/dev/null | head -n 1)"

if [[ -z "$ARCH" ]]; then
  ARCH="$(find /opt/vitis_ai/compiler/arch/DPUCZDX8G \
    -type f -iname arch.json 2>/dev/null | grep -i zcu104 | head -n 1 || true)"
fi

if [[ -z "$ARCH" ]]; then
  echo "Không tự tìm thấy arch.json của ZCU104." >&2
  echo "Hãy chạy: find /opt/vitis_ai/compiler/arch/DPUCZDX8G -name arch.json" >&2
  exit 1
fi

echo "Using arch: $ARCH"
mkdir -p "$OUTPUT_DIR"

vai_c_xir \
  -x "$INPUT_XMODEL" \
  -a "$ARCH" \
  -o "$OUTPUT_DIR" \
  -n "$NET_NAME"

echo "Compiled model nằm trong: $OUTPUT_DIR"
