#!/usr/bin/env python3
"""YOLO11n DPU-friendly standalone cho Vitis AI 3.0.

Tương thích:
- Python 3.7
- PyTorch 1.12.1
- Không phụ thuộc package Ultralytics
- 2 lớp mặc định: bullet, target
- Activation ReLU
- Không C2PSA/Swin
- Trả về 6 raw output tensor để DFL/NMS chạy trên ARM
"""

import math

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Tính padding giữ nguyên kích thước khi stride=1."""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise convolution theo cách Ultralytics xây Detect head."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(
            c1,
            c2,
            k,
            s,
            g=math.gcd(c1, c2),
            d=d,
            act=act,
        )


class Bottleneck(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
    ):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = Conv(hidden, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0,
            )
            for _ in range(n)
        )

    def forward(self, x):
        # split() được dùng thay cho chunk() để thân thiện hơn với NNDCT.
        parts = self.cv1(x).split((self.c, self.c), 1)
        outputs = [parts[0], parts[1]]
        for block in self.m:
            outputs.append(block(outputs[-1]))
        return self.cv2(torch.cat(outputs, 1))


class C3(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
    ):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden,
                    hidden,
                    shortcut,
                    g,
                    k=((1, 1), (3, 3)),
                    e=1.0,
                )
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(
            torch.cat(
                (self.m(self.cv1(x)), self.cv2(x)),
                1,
            )
        )


class C3k(C3):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k=3,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden,
                    hidden,
                    shortcut,
                    g,
                    k=(k, k),
                    e=1.0,
                )
                for _ in range(n)
            )
        )


class C3k2(C2f):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        attn=False,
        g=1,
        shortcut=True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if attn:
            raise ValueError(
                "Bản DPU-friendly không hỗ trợ attention."
            )
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SPPF(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=5,
        n=3,
        shortcut=False,
    ):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1, act=False)
        self.cv2 = Conv(hidden * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=k,
            stride=1,
            padding=k // 2,
        )
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        outputs = [self.cv1(x)]
        for _ in range(self.n):
            outputs.append(self.m(outputs[-1]))
        y = self.cv2(torch.cat(outputs, 1))
        return y + x if self.add else y


class DFL(nn.Module):
    """Giữ module DFL để state_dict khớp checkpoint."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            1,
            1,
            bias=False,
        ).requires_grad_(False)
        values = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(
            values.view(1, c1, 1, 1)
        )
        self.c1 = c1

    def forward(self, x):
        batch, _, anchors = x.shape
        x = x.view(
            batch,
            4,
            self.c1,
            anchors,
        ).transpose(2, 1)
        x = x.softmax(1)
        return self.conv(x).view(batch, 4, anchors)


class RawDetect(nn.Module):
    """Detect head chỉ xuất raw box/class tensors."""

    legacy = False

    def __init__(
        self,
        nc=2,
        reg_max=16,
        ch=(64, 128, 256),
    ):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1),
            )
            for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    DWConv(x, x, 3),
                    Conv(x, c3, 1),
                ),
                nn.Sequential(
                    DWConv(c3, c3, 3),
                    Conv(c3, c3, 1),
                ),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )

        self.dfl = (
            DFL(reg_max)
            if reg_max > 1
            else nn.Identity()
        )

    def forward(self, features):
        outputs = []
        for index in range(self.nl):
            box = self.cv2[index](features[index])
            cls = self.cv3[index](features[index])
            outputs.append(box)
            outputs.append(cls)
        return tuple(outputs)


class YOLO11nZCU104(nn.Module):
    """YOLO11n ReLU, không PSA/Swin, đầu ra raw Detect."""

    def __init__(self, nc=2):
        super().__init__()
        self.nc = nc
        self.names = {
            0: "bullet",
            1: "target",
        }

        # Dùng ModuleList và đúng số thứ tự layer để key state_dict
        # khớp với Ultralytics DetectionModel.
        self.model = nn.ModuleList(
            [
                Conv(3, 16, 3, 2),                       # 0
                Conv(16, 32, 3, 2),                      # 1
                C3k2(32, 64, 1, False, 0.25),            # 2
                Conv(64, 64, 3, 2),                      # 3
                C3k2(64, 128, 1, False, 0.25),           # 4
                Conv(128, 128, 3, 2),                    # 5
                C3k2(128, 128, 1, True, 0.5),            # 6
                Conv(128, 256, 3, 2),                    # 7
                C3k2(256, 256, 1, True, 0.5),            # 8
                SPPF(256, 256, 5),                       # 9
                C3k2(256, 256, 1, True, 0.5),            # 10
                nn.Upsample(
                    size=None,
                    scale_factor=2,
                    mode="nearest",
                ),                                       # 11
                nn.Identity(),                           # 12 concat
                C3k2(384, 128, 1, False, 0.5),           # 13
                nn.Upsample(
                    size=None,
                    scale_factor=2,
                    mode="nearest",
                ),                                       # 14
                nn.Identity(),                           # 15 concat
                C3k2(256, 64, 1, False, 0.5),            # 16 P3
                Conv(64, 64, 3, 2),                      # 17
                nn.Identity(),                           # 18 concat
                C3k2(192, 128, 1, False, 0.5),           # 19 P4
                Conv(128, 128, 3, 2),                    # 20
                nn.Identity(),                           # 21 concat
                C3k2(384, 256, 1, True, 0.5),            # 22 P5
                RawDetect(
                    nc=nc,
                    ch=(64, 128, 256),
                ),                                       # 23
            ]
        )

    def forward(self, x):
        saved = []

        x = self.model[0](x)
        saved.append(x)
        x = self.model[1](x)
        saved.append(x)
        x = self.model[2](x)
        saved.append(x)
        x = self.model[3](x)
        saved.append(x)
        x = self.model[4](x)
        saved.append(x)
        x = self.model[5](x)
        saved.append(x)
        x = self.model[6](x)
        saved.append(x)
        x = self.model[7](x)
        saved.append(x)
        x = self.model[8](x)
        saved.append(x)
        x = self.model[9](x)
        saved.append(x)
        x = self.model[10](x)
        saved.append(x)

        x = self.model[11](x)
        saved.append(x)

        x = torch.cat((x, saved[6]), 1)
        saved.append(x)

        x = self.model[13](x)
        saved.append(x)

        x = self.model[14](x)
        saved.append(x)

        x = torch.cat((x, saved[4]), 1)
        saved.append(x)

        p3 = self.model[16](x)
        saved.append(p3)

        x = self.model[17](p3)
        saved.append(x)

        x = torch.cat((x, saved[13]), 1)
        saved.append(x)

        p4 = self.model[19](x)
        saved.append(p4)

        x = self.model[20](p4)
        saved.append(x)

        x = torch.cat((x, saved[10]), 1)
        saved.append(x)

        p5 = self.model[22](x)
        saved.append(p5)

        return self.model[23]([p3, p4, p5])


def load_standalone_checkpoint(path, map_location="cpu"):
    """Tạo model và strict-load gói state_dict."""
    package = torch.load(path, map_location=map_location)

    if not isinstance(package, dict):
        raise TypeError(
            "Checkpoint standalone phải là dict."
        )

    state_dict = package.get("state_dict", package)
    names = package.get(
        "names",
        {0: "bullet", 1: "target"},
    )
    nc = len(names)

    model = YOLO11nZCU104(nc=nc)
    model.names = names
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, package


if __name__ == "__main__":
    model = YOLO11nZCU104(nc=2).eval()

    print("State dict entries:", len(model.state_dict()))
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(dummy)

    for index, output in enumerate(outputs):
        print(index, tuple(output.shape))
