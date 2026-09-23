# YOLO11 Bullet & Target Detection on FPGA ZCU104

This project implements a customized YOLO11 object detection model for detecting **bullet holes** and **shooting targets**, and deploys the model on the **AMD/Xilinx ZCU104 FPGA** using **Vitis AI 3.0** and the **DPUCZDX8G** accelerator.

The project focuses on converting a YOLO11 model into a DPU-compatible architecture, INT8 quantization, XModel compilation, and inference on the ZCU104 board.

---

## 1. Project Overview

### Objective

The system is designed to detect two object classes:

| Class ID | Class |
|---|---|
| 0 | `bullet` |
| 1 | `target` |

The final deployment pipeline is:

```text
Camera / Image
      |
      v
ARM Pre-processing
      |
      v
640 x 640 INT8 input
      |
      v
+----------------------+
| DPUCZDX8G B4096      |
| YOLO11 XModel        |
+----------------------+
      |
      v
Raw detection tensors
      |
      v
ARM Post-processing
DFL + Decode + NMS
      |
      v
Bullet / Target Detection
```

The current implementation successfully runs the YOLO11 XModel on the ZCU104 DPU.

---

## 2. Hardware Platform

Target board:

```text
AMD/Xilinx ZCU104
Device: Zynq UltraScale+ MPSoC ZU7EV
```

DPU configuration detected on the board:

```text
DPU Architecture : DPUCZDX8G_ISA1_B4096
Number of Cores  : 2
DPU Frequency    : 300 MHz
XRT Frequency    : 300 MHz
VART Runtime     : 3.0.0
```

The project currently uses the official Vitis AI ZCU104 SD image:

```text
xilinx-zcu104-dpu-v2022.2-v3.0.0.img.gz
```

Board environment:

```text
PetaLinux : 2022.2
Linux     : 5.15.36-xilinx-v2022.2
Arch      : aarch64
Vitis AI  : 3.0
```

---

## 3. Development Environment

The model is quantized and compiled inside the Vitis AI 3.0 environment.

Main environment:

```text
Python          : 3.7.12
PyTorch         : 1.12.1
pytorch_nndct   : 3.0.0
Vitis AI        : 3.0
Target DPU      : DPUCZDX8G_ISA1_B4096
```

Quantization is performed on CPU.

CUDA is not required for the current Vitis AI quantization flow.

---

## 4. YOLO11 Architecture Modification

The original YOLO11 architecture contains several operations that are not convenient for deployment on the DPUCZDX8G.

The model was therefore modified for FPGA deployment.

Main modifications:

```text
Original activation:
SiLU
    |
    v
ReLU
```

```text
Original block:
C2PSA
    |
    v
C3k2
```

The final model uses a DPU-friendly architecture based mainly on:

```text
Conv2D
BatchNorm
ReLU
C3k2
SPPF
Concat
Upsample
Detection Head
```

### No-Split C3k2

During the first deployment attempt, `split()` / `chunk()` operations inside C3k2 caused transpose operations to appear in the compiled graph.

This resulted in a Vitis AI compiler error.

Therefore, C3k2 was rewritten without `split()` or `chunk()`.

Instead of:

```python
x = self.cv1(x)
y1, y2 = x.split(...)
```

the implementation uses two independent convolution branches:

```python
y1 = self.cv1a(x)
y2 = self.cv1b(x)
```

The original weights are split along the output-channel dimension and loaded into the two branches.

This implementation successfully compiles for the ZCU104 DPU.

---

## 5. Detection Head

The DPU only performs the neural-network computation.

Post-processing is executed on the ARM processor.

The XModel outputs six raw tensors:

```text
box_s8
cls_s8

box_s16
cls_s16

box_s32
cls_s32
```

Output tensor shapes:

| Output | Shape | Fix Point |
|---|---|---:|
| box_s8 | `[1, 80, 80, 64]` | 3 |
| cls_s8 | `[1, 80, 80, 2]` | 2 |
| box_s16 | `[1, 40, 40, 64]` | 3 |
| cls_s16 | `[1, 40, 40, 2]` | 2 |
| box_s32 | `[1, 20, 20, 64]` | 3 |
| cls_s32 | `[1, 20, 20, 2]` | 2 |

Input tensor:

```text
Shape     : [1, 640, 640, 3]
Data type : INT8
Layout    : NHWC
Fix point : 6
```

Operations such as:

```text
DFL decoding
Bounding-box decoding
Confidence filtering
NMS
```

are performed on the ARM CPU.

---

## 6. Deployment Flow

The complete deployment flow is:

```text
YOLO11 trained model
        |
        v
PyTorch model
        |
        v
Standalone DPU-compatible YOLO11
        |
        v
Vitis AI Quantizer
        |
        v
INT8 XModel
        |
        v
Vitis AI Compiler
        |
        v
DPUCZDX8G B4096 XModel
        |
        v
ZCU104
        |
        +---- ARM preprocessing
        |
        +---- DPU inference
        |
        +---- ARM post-processing
        |
        v
Detection result
```

---

## 7. Compile XModel

The quantized model is compiled using `vai_c_xir`.

Example:

```bash
vai_c_xir \
    -x YOLO11nZCU104_int.xmodel \
    -a arch.json \
    -o compiled_zcu104_nosplit \
    -n yolo11n_bullet_target
```

Successful compilation produces:

```text
compiled_zcu104_nosplit/
└── yolo11n_bullet_target.xmodel
```

Compiler target:

```text
DPUCZDX8G_ISA1_B4096
```

The compiled model contains one main DPU compute subgraph.

---

## 8. Copy Model to ZCU104

Copy the XModel and test scripts to the board:

```bash
scp yolo11n_bullet_target.xmodel \
    root@<ZCU104_IP>:/root/board_compare/bia_zcu104/
```

Example project directory on ZCU104:

```text
/root/board_compare/bia_zcu104/
```

---

## 9. Check DPU

After booting the ZCU104:

```bash
xdputil query
```

The board should report:

```text
DPU Core Count : 2

DPUCZDX8G_1
DPUCZDX8G_2

Architecture:
DPUCZDX8G_ISA1_B4096
```

---

## 10. Check XModel

Example:

```bash
xdputil xmodel yolo11n_bullet_target.xmodel -l
```

Current compiled XModel information:

```text
Workload:
6,411,024,800 operations
```

Memory-related information:

```text
CONST             : 3,170,304 bytes
WORKSPACE         : 13,747,200 bytes
DATA_LOCAL_INPUT  : 1,230,736 bytes
DATA_LOCAL_OUTPUT :   554,400 bytes
```

These values describe the compiled XModel runtime requirements.

They are not equivalent to FPGA LUT/FF/BRAM utilization.

---

## 11. Run XModel Test

Basic DPU test:

```bash
python3 check_yolo11_zcu104.py \
    --model yolo11n_bullet_target.xmodel \
    --image test.jpg
```

Full inference:

```bash
python3 infer_yolo11_zcu104.py \
    --model yolo11n_bullet_target.xmodel \
    --image test.jpg \
    --output result.jpg
```

The pipeline performs:

```text
Image
  |
  v
Resize / preprocess
  |
  v
INT8 conversion
  |
  v
DPU inference
  |
  v
Dequantization
  |
  v
DFL decoding
  |
  v
NMS
  |
  v
Draw detection
```

---

## 12. Performance

The YOLO11 XModel has been successfully executed on the ZCU104.

Measured DPU-only inference time is approximately:

```text
12.5 - 13.7 ms / image
```

Equivalent DPU-only throughput is approximately:

```text
73 - 80 FPS
```

> These values represent DPU execution time only and do not include the complete camera, preprocessing, post-processing, display, or file-I/O pipeline.

One Vitis AI profiling run reported:

```text
Workload       : 6.411 GOP
DPU time       : 12.471 ms
Memory I/O     : 41.272 MB
Memory BW      : 3352.911 MB/s
```

---

## 13. FPGA Resource Notes

The XModel itself does not directly allocate a specific number of FPGA LUTs or registers.

The FPGA resources are used to implement the DPU hardware architecture in the FPGA bitstream.

The XModel is executed as instructions on the existing DPU.

Therefore:

```text
FPGA Bitstream
      |
      v
DPUCZDX8G B4096
      |
      +---- LUT
      +---- Register
      +---- DSP
      +---- BRAM / URAM
      |
      v
YOLO11 XModel instructions
```

The resource utilization of the DPU hardware and the runtime requirements of the XModel should therefore be evaluated separately.

---

## 14. Current Project Status

Current status:

```text
[OK] YOLO11 custom architecture
[OK] DPU-compatible model
[OK] INT8 quantization
[OK] XModel export
[OK] Vitis AI compilation
[OK] ZCU104 DPU execution
[OK] Image inference
[OK] DPU profiling
[ ] USB camera input
[ ] Real-time camera inference
[ ] Bullet counting
[ ] Shooting target scoring
[ ] Real-time visualization
```

---

## 15. Planned Real-Time Camera System

The next development stage is real-time bullet detection using a USB camera.

Planned system:

```text
USB Camera
     |
     v
V4L2 / OpenCV
     |
     v
ARM preprocessing
     |
     v
YOLO11 XModel
     |
     v
DPUCZDX8G
     |
     v
ARM post-processing
     |
     +---- Detect target
     |
     +---- Detect bullet
     |
     v
Bullet position / score
```

The system is intended to operate headlessly, so DisplayPort is not required for inference.

Results can be:

```text
saved to image/video
streamed to another computer
or displayed through DisplayPort
```

---

## 16. Project Structure

An example repository structure is:

```text
biaban_yolov11/
│
├── README.md
│
├── model/
│   ├── yolo11n_bullet_target.xmodel
│   └── ...
│
├── quant/
│   ├── quantization scripts
│   └── quantize_result/
│
├── compile/
│   ├── arch.json
│   └── compiled model
│
├── zcu104/
│   ├── check_yolo11_zcu104.py
│   ├── infer_yolo11_zcu104.py
│   └── ...
│
├── test_images/
│   └── ...
│
└── results/
    └── ...
```

> The actual directory structure may change during development.

---

## 17. Backup Repository

Clone this repository:

```bash
git clone https://github.com/dgvu/biaban_yolov11.git
cd biaban_yolov11
```

Update the repository:

```bash
git add .
git commit -m "Update YOLO11 ZCU104 deployment"
git push origin main
```

---

## 18. Technologies

```text
Python
PyTorch
Ultralytics YOLO11
Vitis AI 3.0
PyTorch NNDCT
XIR
VART
OpenCV
PetaLinux
AMD/Xilinx ZCU104
DPUCZDX8G
```

---

## 19. Future Work

Future development will focus on:

- Real-time USB camera inference on ZCU104
- Automatic bullet-hole detection
- Target detection and localization
- Bullet counting
- Shooting-score calculation
- Performance optimization between ARM and DPU
- End-to-end FPS measurement
- FPGA/DPU resource analysis
- Comparison between FP32, INT8 and deployed XModel results

---

## Author

**dgvu**

Repository:

```text
https://github.com/dgvu/biaban_yolov11
```

---

## Notes

This repository is currently used for research and development of a YOLO11-based FPGA object detection system on the AMD/Xilinx ZCU104 platform.

The implementation is still under development.
