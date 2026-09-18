import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse
from Models.ImEncoderUtil import *
from Models.NetUtil import *

class ImEncoder(nn.Module):
    def __init__(self, in_channels, base_planes, fpn_out=256):
        super().__init__()
        self.backbone = ResNetBackbone(BasicBlockWithDWT, [2, 2, 2, 2], in_channels=in_channels, base_planes=base_planes, do_dwt=True)
        self.fpn = FPNNeck(in_channels=[base_planes, base_planes*2, base_planes*4, base_planes*8], out_channels=fpn_out)

    def forward(self, x):
        c_feats = self.backbone(x)      # (c2, c3, c4, c5)
        p_feats = self.fpn(c_feats)     # [p5, p4, p3, p2]
        return p_feats

class BasicBlockWithDWT(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        if self.downsample is not None:
            self.dwt = DWTForward(J=1, wave='haar')
            self.dwtconv = nn.Sequential(
                    nn.Conv2d(4*inplanes, planes, kernel_size=3,stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
    def _transformer(self, dwt_yl, dwt_yh):
        list_tensor = []
        a = dwt_yh[0]
        list_tensor.append(dwt_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            dwt_yl, dwt_yh = self.dwt(x)
            dwtt = self.dwtconv(self._transformer(dwt_yl, dwt_yh))
        out = self.relu(self.bn1(self.conv1(x)))
        if self.downsample is not None:
            out = out + dwtt
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """
    Configurable ResNet backbone producing C2, C3, C4, C5 for feature pyramids.

    Args:
        block: BasicBlock / Bottleneck
        layers: list, e.g., [3,4,6,3] for ResNet-50
        in_channels: input channels (default 3)
        base_planes: base channel count (default 64)
    """
    def __init__(self, block, layers, in_channels=3, base_planes=64, block_strides=[1,2,2,2], do_dwt=False, norm_cfg=None):
        super().__init__()
        self.inplanes = base_planes
        self.layers = nn.ModuleList()
        # ---- Stem ----
        self.conv1 = nn.Conv2d(in_channels, base_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, base_planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.do_dwt = do_dwt
        if self.do_dwt:
            self.dwt = DWTForward(J=1, wave='haar')
            self.dwtconv = nn.Sequential(
                    nn.Conv2d(4*base_planes, base_planes, kernel_size=3,stride=1, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True))

        # ---- Stages: C2 to C5 ----
        for i in range(4):
            self.layers.append(self._make_layer(block, base_planes*2**i, layers[i], block_strides[i]))

        # Kaiming Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _transformer(self, dwt_yl, dwt_yh):
        list_tensor = []
        a = dwt_yh[0]
        list_tensor.append(dwt_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)

    def _make_layer(self, block, planes, blocks, stride=1, norm_cfg=None):
        downsample = None

        # Whether we need a projection
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_cfg)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))  
        if self.do_dwt:
            dwt_yl, dwt_yh = self.dwt(x)
            dwtt = self.dwtconv(self._transformer(dwt_yl, dwt_yh))
        x = self.maxpool(x)    
        if self.do_dwt: 
            x = x + dwtt               
        outputs = []
        for layer in self.layers:
            x = layer(x)     
            outputs.append(x)
        return outputs

class FPNNeck(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        assert len(in_channels) == 4, "FPN expects 4 input feature levels (C2-C5)"
        c2, c3, c4, c5 = in_channels
        # 1x1 lateral convolutions
        self.lateral_c2 = nn.Conv2d(c2, out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(c5, out_channels, kernel_size=1)

        # 3x3 output convolutions
        self.out_c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_c3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_c4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_c5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # dynamic filter
        self.df = DynamicFilter(dim=out_channels, weight_resize=True)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # lateral features
        p5 = self.lateral_c5(c5); p5 = self.df(p5)
        p4 = self.lateral_c4(c4); p4 = self.df(p4)
        p3 = self.lateral_c3(c3); p3 = self.df(p3)
        p2 = self.lateral_c2(c2); p2 = self.df(p2)

        # top-down pathway
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        # smoothing
        p5 = self.out_c5(p5)
        p4 = self.out_c4(p4)
        p3 = self.out_c3(p3)
        p2 = self.out_c2(p2)

        return [p5, p4, p3, p2]