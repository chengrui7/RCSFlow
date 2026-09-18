# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from Models.NetUtil import *
from Models.OccEncoderUtil import *

class OccResNet3D(nn.Module):
    def __init__(self,
                 depth,
                 block_inplanes=[64, 128, 256, 512],
                 block_strides=[1, 2, 2, 2],
                 out_indices=(0, 1, 2, 3),
                 n_input_channels=3,
                 shortcut_type='B',
                 norm_cfg=dict(type='GN', requires_grad=True),
                 widen_factor=1.0):
        super().__init__()
        
        layer_metas = {
            10: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }
        
        if depth in [10, 18, 34]:
            block = BasicBlock
        else:
            assert depth in [50, 101]
            block = Bottleneck
        
        layers = layer_metas[depth]
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.out_indices = out_indices
        
        # replace the first several downsampling layers with the channel-squeeze layers
        self.input_proj = nn.Sequential(
            nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), bias=False),
            build_norm_layer(norm_cfg, self.in_planes)[1],
            nn.ReLU(inplace=True),
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(block_inplanes)):
            self.layers.append(self._make_layer(block, block_inplanes[i], layers[i], 
                                shortcut_type, block_strides[i], norm_cfg=norm_cfg))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm_cfg=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  norm_cfg=norm_cfg))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x)
        res = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            
            if index in self.out_indices:
                res.append(x)
        return res
    

class FPN3D(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """
    def __init__(self,
                 in_channels=[80, 160, 320, 640],
                 out_channels=256,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 with_cp=False,
                 upsample_cfg=dict(mode='trilinear'),
                 init_cfg=None):
        super(FPN3D, self).__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        
        self.num_out = len(self.in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.num_out):
            l_conv = conv_module_3d(in_channels=in_channels[i], out_channels=out_channels, kernel_size=1, padding=0,
                                    norm_cfg=norm_cfg, bias=False, inplace=True) 
            fpn_conv = conv_module_3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                    norm_cfg=norm_cfg, bias=False, inplace=True) 
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if self.with_cp:
                lateral_i = torch.utils.checkpoint.checkpoint(lateral_conv, inputs[i])
            else:
                lateral_i = lateral_conv(inputs[i])
            laterals.append(lateral_i)

        # build down-top path
        for i in range(self.num_out - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], 
                    size=prev_shape, align_corners=False, **self.upsample_cfg)
        
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(self.num_out)
        # ]
        
        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            if self.with_cp:
                out_i = torch.utils.checkpoint.checkpoint(fpn_conv, laterals[i])
            else:
                out_i = fpn_conv(laterals[i])
            outs.append(out_i)
        
        return outs