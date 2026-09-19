# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class YOLO11nZCU104(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(YOLO11nZCU104, self).__init__()
        self.module_0 = py_nndct.nn.Input() #YOLO11nZCU104::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[0]/Conv2d[conv]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[0]/ReLU[act]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[1]/Conv2d[conv]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[1]/ReLU[act]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Conv[cv1a]/Conv2d[conv]/input.15
        self.module_6 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Conv[cv1a]/ReLU[act]/15033
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Conv[cv1b]/Conv2d[conv]/input.19
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Conv[cv1b]/ReLU[act]/input.23
        self.module_9 = py_nndct.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.25
        self.module_10 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.29
        self.module_11 = py_nndct.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.31
        self.module_12 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/15111
        self.module_13 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Bottleneck[m]/ModuleList[0]/15113
        self.module_14 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/input.35
        self.module_15 = py_nndct.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Conv[cv2]/Conv2d[conv]/input.37
        self.module_16 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[2]/Conv[cv2]/ReLU[act]/input.41
        self.module_17 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[3]/Conv2d[conv]/input.43
        self.module_18 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[3]/ReLU[act]/input.47
        self.module_19 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Conv[cv1a]/Conv2d[conv]/input.49
        self.module_20 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Conv[cv1a]/ReLU[act]/15194
        self.module_21 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Conv[cv1b]/Conv2d[conv]/input.53
        self.module_22 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Conv[cv1b]/ReLU[act]/input.57
        self.module_23 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.59
        self.module_24 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.63
        self.module_25 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.65
        self.module_26 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/15272
        self.module_27 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Bottleneck[m]/ModuleList[0]/15274
        self.module_28 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/input.69
        self.module_29 = py_nndct.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Conv[cv2]/Conv2d[conv]/input.71
        self.module_30 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[4]/Conv[cv2]/ReLU[act]/input.75
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[5]/Conv2d[conv]/input.77
        self.module_32 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[5]/ReLU[act]/input.81
        self.module_33 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/Conv[cv1a]/Conv2d[conv]/input.83
        self.module_34 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/Conv[cv1a]/ReLU[act]/15355
        self.module_35 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/Conv[cv1b]/Conv2d[conv]/input.87
        self.module_36 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/Conv[cv1b]/ReLU[act]/input.91
        self.module_37 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.93
        self.module_38 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.97
        self.module_39 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.99
        self.module_40 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.103
        self.module_41 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.105
        self.module_42 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/15459
        self.module_43 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/input.109
        self.module_44 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.111
        self.module_45 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/ReLU[act]/input.115
        self.module_46 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.117
        self.module_47 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/ReLU[act]/15513
        self.module_48 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/15515
        self.module_49 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.121
        self.module_50 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/15541
        self.module_51 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/input.125
        self.module_52 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/input.127
        self.module_53 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/C3k[m]/ModuleList[0]/Conv[cv3]/ReLU[act]/15570
        self.module_54 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/input.131
        self.module_55 = py_nndct.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/Conv[cv2]/Conv2d[conv]/input.133
        self.module_56 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[6]/Conv[cv2]/ReLU[act]/input.137
        self.module_57 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[7]/Conv2d[conv]/input.139
        self.module_58 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[7]/ReLU[act]/input.143
        self.module_59 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/Conv[cv1a]/Conv2d[conv]/input.145
        self.module_60 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/Conv[cv1a]/ReLU[act]/15651
        self.module_61 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/Conv[cv1b]/Conv2d[conv]/input.149
        self.module_62 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/Conv[cv1b]/ReLU[act]/input.153
        self.module_63 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.155
        self.module_64 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.159
        self.module_65 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.161
        self.module_66 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.165
        self.module_67 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.167
        self.module_68 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/15755
        self.module_69 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/input.171
        self.module_70 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.173
        self.module_71 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/ReLU[act]/input.177
        self.module_72 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.179
        self.module_73 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/ReLU[act]/15809
        self.module_74 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/15811
        self.module_75 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.183
        self.module_76 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/15837
        self.module_77 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/input.187
        self.module_78 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/input.189
        self.module_79 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/C3k[m]/ModuleList[0]/Conv[cv3]/ReLU[act]/15866
        self.module_80 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/input.193
        self.module_81 = py_nndct.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/Conv[cv2]/Conv2d[conv]/input.195
        self.module_82 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[8]/Conv[cv2]/ReLU[act]/input.199
        self.module_83 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/Conv[cv1]/Conv2d[conv]/input.201
        self.module_84 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/MaxPool2d[m]/15935
        self.module_85 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/MaxPool2d[m]/15949
        self.module_86 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/MaxPool2d[m]/15963
        self.module_87 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/input.203
        self.module_88 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/Conv[cv2]/Conv2d[conv]/input.205
        self.module_89 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/SPPF[model]/ModuleList[9]/Conv[cv2]/ReLU[act]/input.209
        self.module_90 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/Conv[cv1a]/Conv2d[conv]/input.211
        self.module_91 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/Conv[cv1a]/ReLU[act]/16018
        self.module_92 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/Conv[cv1b]/Conv2d[conv]/input.215
        self.module_93 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/Conv[cv1b]/ReLU[act]/input.219
        self.module_94 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.221
        self.module_95 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.225
        self.module_96 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.227
        self.module_97 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.231
        self.module_98 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.233
        self.module_99 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/16122
        self.module_100 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/input.237
        self.module_101 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.239
        self.module_102 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/ReLU[act]/input.243
        self.module_103 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.245
        self.module_104 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/ReLU[act]/16176
        self.module_105 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/16178
        self.module_106 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.249
        self.module_107 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/16204
        self.module_108 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/input.253
        self.module_109 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/input.255
        self.module_110 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/C3k[m]/ModuleList[0]/Conv[cv3]/ReLU[act]/16233
        self.module_111 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/input.259
        self.module_112 = py_nndct.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/Conv[cv2]/Conv2d[conv]/input.261
        self.module_113 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[10]/Conv[cv2]/ReLU[act]/input.265
        self.module_114 = py_nndct.nn.Interpolate() #YOLO11nZCU104::YOLO11nZCU104/Upsample[model]/ModuleList[11]/16267
        self.module_115 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/input.267
        self.module_116 = py_nndct.nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Conv[cv1a]/Conv2d[conv]/input.269
        self.module_117 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Conv[cv1a]/ReLU[act]/16296
        self.module_118 = py_nndct.nn.Conv2d(in_channels=384, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Conv[cv1b]/Conv2d[conv]/input.273
        self.module_119 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Conv[cv1b]/ReLU[act]/input.277
        self.module_120 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.279
        self.module_121 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.283
        self.module_122 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.285
        self.module_123 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/16374
        self.module_124 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Bottleneck[m]/ModuleList[0]/16376
        self.module_125 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/input.289
        self.module_126 = py_nndct.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Conv[cv2]/Conv2d[conv]/input.291
        self.module_127 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[13]/Conv[cv2]/ReLU[act]/input.295
        self.module_128 = py_nndct.nn.Interpolate() #YOLO11nZCU104::YOLO11nZCU104/Upsample[model]/ModuleList[14]/16410
        self.module_129 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/input.297
        self.module_130 = py_nndct.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Conv[cv1a]/Conv2d[conv]/input.299
        self.module_131 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Conv[cv1a]/ReLU[act]/16439
        self.module_132 = py_nndct.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Conv[cv1b]/Conv2d[conv]/input.303
        self.module_133 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Conv[cv1b]/ReLU[act]/input.307
        self.module_134 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.309
        self.module_135 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.313
        self.module_136 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.315
        self.module_137 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/16517
        self.module_138 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Bottleneck[m]/ModuleList[0]/16519
        self.module_139 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/input.319
        self.module_140 = py_nndct.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Conv[cv2]/Conv2d[conv]/input.321
        self.module_141 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[16]/Conv[cv2]/ReLU[act]/input.325
        self.module_142 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[17]/Conv2d[conv]/input.327
        self.module_143 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[17]/ReLU[act]/16574
        self.module_144 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/input.331
        self.module_145 = py_nndct.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Conv[cv1a]/Conv2d[conv]/input.333
        self.module_146 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Conv[cv1a]/ReLU[act]/16603
        self.module_147 = py_nndct.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Conv[cv1b]/Conv2d[conv]/input.337
        self.module_148 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Conv[cv1b]/ReLU[act]/input.341
        self.module_149 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.343
        self.module_150 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Bottleneck[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.347
        self.module_151 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.349
        self.module_152 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Bottleneck[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/16681
        self.module_153 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Bottleneck[m]/ModuleList[0]/16683
        self.module_154 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/input.353
        self.module_155 = py_nndct.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Conv[cv2]/Conv2d[conv]/input.355
        self.module_156 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[19]/Conv[cv2]/ReLU[act]/input.359
        self.module_157 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[20]/Conv2d[conv]/input.361
        self.module_158 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/Conv[model]/ModuleList[20]/ReLU[act]/16738
        self.module_159 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/input.365
        self.module_160 = py_nndct.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/Conv[cv1a]/Conv2d[conv]/input.367
        self.module_161 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/Conv[cv1a]/ReLU[act]/16767
        self.module_162 = py_nndct.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/Conv[cv1b]/Conv2d[conv]/input.371
        self.module_163 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/Conv[cv1b]/ReLU[act]/input.375
        self.module_164 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Conv[cv1]/Conv2d[conv]/input.377
        self.module_165 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Conv[cv1]/ReLU[act]/input.381
        self.module_166 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.383
        self.module_167 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv1]/ReLU[act]/input.387
        self.module_168 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.389
        self.module_169 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/Conv[cv2]/ReLU[act]/16871
        self.module_170 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[0]/input.393
        self.module_171 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.395
        self.module_172 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv1]/ReLU[act]/input.399
        self.module_173 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.401
        self.module_174 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/Conv[cv2]/ReLU[act]/16925
        self.module_175 = py_nndct.nn.Add() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Sequential[m]/Bottleneck[1]/16927
        self.module_176 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Conv[cv2]/Conv2d[conv]/input.405
        self.module_177 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Conv[cv2]/ReLU[act]/16953
        self.module_178 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/input.409
        self.module_179 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/input.411
        self.module_180 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/C3k[m]/ModuleList[0]/Conv[cv3]/ReLU[act]/16982
        self.module_181 = py_nndct.nn.Cat() #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/input.415
        self.module_182 = py_nndct.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/Conv[cv2]/Conv2d[conv]/input.417
        self.module_183 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/C3k2[model]/ModuleList[22]/Conv[cv2]/ReLU[act]/input.493
        self.module_184 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[0]/Conv[0]/Conv2d[conv]/input.421
        self.module_185 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[0]/Conv[0]/ReLU[act]/input.425
        self.module_186 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[0]/Conv[1]/Conv2d[conv]/input.427
        self.module_187 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[0]/Conv[1]/ReLU[act]/input.431
        self.module_188 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[0]/Conv2d[2]/17082
        self.module_189 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[0]/DWConv[0]/Conv2d[conv]/input.433
        self.module_190 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[0]/DWConv[0]/ReLU[act]/input.437
        self.module_191 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[0]/Conv[1]/Conv2d[conv]/input.439
        self.module_192 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[0]/Conv[1]/ReLU[act]/input.443
        self.module_193 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[1]/DWConv[0]/Conv2d[conv]/input.445
        self.module_194 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[1]/DWConv[0]/ReLU[act]/input.449
        self.module_195 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[1]/Conv[1]/Conv2d[conv]/input.451
        self.module_196 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Sequential[1]/Conv[1]/ReLU[act]/input.455
        self.module_197 = py_nndct.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[0]/Conv2d[2]/17205
        self.module_198 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[1]/Conv[0]/Conv2d[conv]/input.457
        self.module_199 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[1]/Conv[0]/ReLU[act]/input.461
        self.module_200 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[1]/Conv[1]/Conv2d[conv]/input.463
        self.module_201 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[1]/Conv[1]/ReLU[act]/input.467
        self.module_202 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[1]/Conv2d[2]/17276
        self.module_203 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[0]/DWConv[0]/Conv2d[conv]/input.469
        self.module_204 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[0]/DWConv[0]/ReLU[act]/input.473
        self.module_205 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[0]/Conv[1]/Conv2d[conv]/input.475
        self.module_206 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[0]/Conv[1]/ReLU[act]/input.479
        self.module_207 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[1]/DWConv[0]/Conv2d[conv]/input.481
        self.module_208 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[1]/DWConv[0]/ReLU[act]/input.485
        self.module_209 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[1]/Conv[1]/Conv2d[conv]/input.487
        self.module_210 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Sequential[1]/Conv[1]/ReLU[act]/input.491
        self.module_211 = py_nndct.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[1]/Conv2d[2]/17399
        self.module_212 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[2]/Conv[0]/Conv2d[conv]/input.495
        self.module_213 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[2]/Conv[0]/ReLU[act]/input.499
        self.module_214 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[2]/Conv[1]/Conv2d[conv]/input.501
        self.module_215 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[2]/Conv[1]/ReLU[act]/input.505
        self.module_216 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv2]/ModuleList[2]/Conv2d[2]/17470
        self.module_217 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=256, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[0]/DWConv[0]/Conv2d[conv]/input.507
        self.module_218 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[0]/DWConv[0]/ReLU[act]/input.511
        self.module_219 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[0]/Conv[1]/Conv2d[conv]/input.513
        self.module_220 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[0]/Conv[1]/ReLU[act]/input.517
        self.module_221 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[1]/DWConv[0]/Conv2d[conv]/input.519
        self.module_222 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[1]/DWConv[0]/ReLU[act]/input.523
        self.module_223 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[1]/Conv[1]/Conv2d[conv]/input.525
        self.module_224 = py_nndct.nn.ReLU(inplace=False) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Sequential[1]/Conv[1]/ReLU[act]/input
        self.module_225 = py_nndct.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #YOLO11nZCU104::YOLO11nZCU104/RawDetect[model]/ModuleList[23]/Sequential[cv3]/ModuleList[2]/Conv2d[2]/17593

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_7 = self.module_7(output_module_0)
        output_module_7 = self.module_8(output_module_7)
        output_module_9 = self.module_9(output_module_7)
        output_module_9 = self.module_10(output_module_9)
        output_module_9 = self.module_11(output_module_9)
        output_module_9 = self.module_12(output_module_9)
        output_module_13 = self.module_13(input=output_module_7, other=output_module_9, alpha=1)
        output_module_5 = self.module_14(dim=1, tensors=[output_module_5,output_module_7,output_module_13])
        output_module_5 = self.module_15(output_module_5)
        output_module_5 = self.module_16(output_module_5)
        output_module_5 = self.module_17(output_module_5)
        output_module_5 = self.module_18(output_module_5)
        output_module_19 = self.module_19(output_module_5)
        output_module_19 = self.module_20(output_module_19)
        output_module_21 = self.module_21(output_module_5)
        output_module_21 = self.module_22(output_module_21)
        output_module_23 = self.module_23(output_module_21)
        output_module_23 = self.module_24(output_module_23)
        output_module_23 = self.module_25(output_module_23)
        output_module_23 = self.module_26(output_module_23)
        output_module_27 = self.module_27(input=output_module_21, other=output_module_23, alpha=1)
        output_module_19 = self.module_28(dim=1, tensors=[output_module_19,output_module_21,output_module_27])
        output_module_19 = self.module_29(output_module_19)
        output_module_19 = self.module_30(output_module_19)
        output_module_31 = self.module_31(output_module_19)
        output_module_31 = self.module_32(output_module_31)
        output_module_33 = self.module_33(output_module_31)
        output_module_33 = self.module_34(output_module_33)
        output_module_35 = self.module_35(output_module_31)
        output_module_35 = self.module_36(output_module_35)
        output_module_37 = self.module_37(output_module_35)
        output_module_37 = self.module_38(output_module_37)
        output_module_39 = self.module_39(output_module_37)
        output_module_39 = self.module_40(output_module_39)
        output_module_39 = self.module_41(output_module_39)
        output_module_39 = self.module_42(output_module_39)
        output_module_43 = self.module_43(input=output_module_37, other=output_module_39, alpha=1)
        output_module_44 = self.module_44(output_module_43)
        output_module_44 = self.module_45(output_module_44)
        output_module_44 = self.module_46(output_module_44)
        output_module_44 = self.module_47(output_module_44)
        output_module_48 = self.module_48(input=output_module_43, other=output_module_44, alpha=1)
        output_module_49 = self.module_49(output_module_35)
        output_module_49 = self.module_50(output_module_49)
        output_module_48 = self.module_51(dim=1, tensors=[output_module_48,output_module_49])
        output_module_48 = self.module_52(output_module_48)
        output_module_48 = self.module_53(output_module_48)
        output_module_33 = self.module_54(dim=1, tensors=[output_module_33,output_module_35,output_module_48])
        output_module_33 = self.module_55(output_module_33)
        output_module_33 = self.module_56(output_module_33)
        output_module_57 = self.module_57(output_module_33)
        output_module_57 = self.module_58(output_module_57)
        output_module_59 = self.module_59(output_module_57)
        output_module_59 = self.module_60(output_module_59)
        output_module_61 = self.module_61(output_module_57)
        output_module_61 = self.module_62(output_module_61)
        output_module_63 = self.module_63(output_module_61)
        output_module_63 = self.module_64(output_module_63)
        output_module_65 = self.module_65(output_module_63)
        output_module_65 = self.module_66(output_module_65)
        output_module_65 = self.module_67(output_module_65)
        output_module_65 = self.module_68(output_module_65)
        output_module_69 = self.module_69(input=output_module_63, other=output_module_65, alpha=1)
        output_module_70 = self.module_70(output_module_69)
        output_module_70 = self.module_71(output_module_70)
        output_module_70 = self.module_72(output_module_70)
        output_module_70 = self.module_73(output_module_70)
        output_module_74 = self.module_74(input=output_module_69, other=output_module_70, alpha=1)
        output_module_75 = self.module_75(output_module_61)
        output_module_75 = self.module_76(output_module_75)
        output_module_74 = self.module_77(dim=1, tensors=[output_module_74,output_module_75])
        output_module_74 = self.module_78(output_module_74)
        output_module_74 = self.module_79(output_module_74)
        output_module_59 = self.module_80(dim=1, tensors=[output_module_59,output_module_61,output_module_74])
        output_module_59 = self.module_81(output_module_59)
        output_module_59 = self.module_82(output_module_59)
        output_module_59 = self.module_83(output_module_59)
        output_module_84 = self.module_84(output_module_59)
        output_module_85 = self.module_85(output_module_84)
        output_module_86 = self.module_86(output_module_85)
        output_module_87 = self.module_87(dim=1, tensors=[output_module_59,output_module_84,output_module_85,output_module_86])
        output_module_87 = self.module_88(output_module_87)
        output_module_87 = self.module_89(output_module_87)
        output_module_90 = self.module_90(output_module_87)
        output_module_90 = self.module_91(output_module_90)
        output_module_92 = self.module_92(output_module_87)
        output_module_92 = self.module_93(output_module_92)
        output_module_94 = self.module_94(output_module_92)
        output_module_94 = self.module_95(output_module_94)
        output_module_96 = self.module_96(output_module_94)
        output_module_96 = self.module_97(output_module_96)
        output_module_96 = self.module_98(output_module_96)
        output_module_96 = self.module_99(output_module_96)
        output_module_100 = self.module_100(input=output_module_94, other=output_module_96, alpha=1)
        output_module_101 = self.module_101(output_module_100)
        output_module_101 = self.module_102(output_module_101)
        output_module_101 = self.module_103(output_module_101)
        output_module_101 = self.module_104(output_module_101)
        output_module_105 = self.module_105(input=output_module_100, other=output_module_101, alpha=1)
        output_module_106 = self.module_106(output_module_92)
        output_module_106 = self.module_107(output_module_106)
        output_module_105 = self.module_108(dim=1, tensors=[output_module_105,output_module_106])
        output_module_105 = self.module_109(output_module_105)
        output_module_105 = self.module_110(output_module_105)
        output_module_90 = self.module_111(dim=1, tensors=[output_module_90,output_module_92,output_module_105])
        output_module_90 = self.module_112(output_module_90)
        output_module_90 = self.module_113(output_module_90)
        output_module_114 = self.module_114(input=output_module_90, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_114 = self.module_115(dim=1, tensors=[output_module_114,output_module_33])
        output_module_116 = self.module_116(output_module_114)
        output_module_116 = self.module_117(output_module_116)
        output_module_118 = self.module_118(output_module_114)
        output_module_118 = self.module_119(output_module_118)
        output_module_120 = self.module_120(output_module_118)
        output_module_120 = self.module_121(output_module_120)
        output_module_120 = self.module_122(output_module_120)
        output_module_120 = self.module_123(output_module_120)
        output_module_124 = self.module_124(input=output_module_118, other=output_module_120, alpha=1)
        output_module_116 = self.module_125(dim=1, tensors=[output_module_116,output_module_118,output_module_124])
        output_module_116 = self.module_126(output_module_116)
        output_module_116 = self.module_127(output_module_116)
        output_module_128 = self.module_128(input=output_module_116, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_128 = self.module_129(dim=1, tensors=[output_module_128,output_module_19])
        output_module_130 = self.module_130(output_module_128)
        output_module_130 = self.module_131(output_module_130)
        output_module_132 = self.module_132(output_module_128)
        output_module_132 = self.module_133(output_module_132)
        output_module_134 = self.module_134(output_module_132)
        output_module_134 = self.module_135(output_module_134)
        output_module_134 = self.module_136(output_module_134)
        output_module_134 = self.module_137(output_module_134)
        output_module_138 = self.module_138(input=output_module_132, other=output_module_134, alpha=1)
        output_module_130 = self.module_139(dim=1, tensors=[output_module_130,output_module_132,output_module_138])
        output_module_130 = self.module_140(output_module_130)
        output_module_130 = self.module_141(output_module_130)
        output_module_142 = self.module_142(output_module_130)
        output_module_142 = self.module_143(output_module_142)
        output_module_142 = self.module_144(dim=1, tensors=[output_module_142,output_module_116])
        output_module_145 = self.module_145(output_module_142)
        output_module_145 = self.module_146(output_module_145)
        output_module_147 = self.module_147(output_module_142)
        output_module_147 = self.module_148(output_module_147)
        output_module_149 = self.module_149(output_module_147)
        output_module_149 = self.module_150(output_module_149)
        output_module_149 = self.module_151(output_module_149)
        output_module_149 = self.module_152(output_module_149)
        output_module_153 = self.module_153(input=output_module_147, other=output_module_149, alpha=1)
        output_module_145 = self.module_154(dim=1, tensors=[output_module_145,output_module_147,output_module_153])
        output_module_145 = self.module_155(output_module_145)
        output_module_145 = self.module_156(output_module_145)
        output_module_157 = self.module_157(output_module_145)
        output_module_157 = self.module_158(output_module_157)
        output_module_157 = self.module_159(dim=1, tensors=[output_module_157,output_module_90])
        output_module_160 = self.module_160(output_module_157)
        output_module_160 = self.module_161(output_module_160)
        output_module_162 = self.module_162(output_module_157)
        output_module_162 = self.module_163(output_module_162)
        output_module_164 = self.module_164(output_module_162)
        output_module_164 = self.module_165(output_module_164)
        output_module_166 = self.module_166(output_module_164)
        output_module_166 = self.module_167(output_module_166)
        output_module_166 = self.module_168(output_module_166)
        output_module_166 = self.module_169(output_module_166)
        output_module_170 = self.module_170(input=output_module_164, other=output_module_166, alpha=1)
        output_module_171 = self.module_171(output_module_170)
        output_module_171 = self.module_172(output_module_171)
        output_module_171 = self.module_173(output_module_171)
        output_module_171 = self.module_174(output_module_171)
        output_module_175 = self.module_175(input=output_module_170, other=output_module_171, alpha=1)
        output_module_176 = self.module_176(output_module_162)
        output_module_176 = self.module_177(output_module_176)
        output_module_175 = self.module_178(dim=1, tensors=[output_module_175,output_module_176])
        output_module_175 = self.module_179(output_module_175)
        output_module_175 = self.module_180(output_module_175)
        output_module_160 = self.module_181(dim=1, tensors=[output_module_160,output_module_162,output_module_175])
        output_module_160 = self.module_182(output_module_160)
        output_module_160 = self.module_183(output_module_160)
        output_module_184 = self.module_184(output_module_130)
        output_module_184 = self.module_185(output_module_184)
        output_module_184 = self.module_186(output_module_184)
        output_module_184 = self.module_187(output_module_184)
        output_module_184 = self.module_188(output_module_184)
        output_module_189 = self.module_189(output_module_130)
        output_module_189 = self.module_190(output_module_189)
        output_module_189 = self.module_191(output_module_189)
        output_module_189 = self.module_192(output_module_189)
        output_module_189 = self.module_193(output_module_189)
        output_module_189 = self.module_194(output_module_189)
        output_module_189 = self.module_195(output_module_189)
        output_module_189 = self.module_196(output_module_189)
        output_module_189 = self.module_197(output_module_189)
        output_module_198 = self.module_198(output_module_145)
        output_module_198 = self.module_199(output_module_198)
        output_module_198 = self.module_200(output_module_198)
        output_module_198 = self.module_201(output_module_198)
        output_module_198 = self.module_202(output_module_198)
        output_module_203 = self.module_203(output_module_145)
        output_module_203 = self.module_204(output_module_203)
        output_module_203 = self.module_205(output_module_203)
        output_module_203 = self.module_206(output_module_203)
        output_module_203 = self.module_207(output_module_203)
        output_module_203 = self.module_208(output_module_203)
        output_module_203 = self.module_209(output_module_203)
        output_module_203 = self.module_210(output_module_203)
        output_module_203 = self.module_211(output_module_203)
        output_module_212 = self.module_212(output_module_160)
        output_module_212 = self.module_213(output_module_212)
        output_module_212 = self.module_214(output_module_212)
        output_module_212 = self.module_215(output_module_212)
        output_module_212 = self.module_216(output_module_212)
        output_module_217 = self.module_217(output_module_160)
        output_module_217 = self.module_218(output_module_217)
        output_module_217 = self.module_219(output_module_217)
        output_module_217 = self.module_220(output_module_217)
        output_module_217 = self.module_221(output_module_217)
        output_module_217 = self.module_222(output_module_217)
        output_module_217 = self.module_223(output_module_217)
        output_module_217 = self.module_224(output_module_217)
        output_module_217 = self.module_225(output_module_217)
        return (output_module_184,output_module_189,output_module_198,output_module_203,output_module_212,output_module_217)
