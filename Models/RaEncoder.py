import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv

from Models.RaEncoderUtil import *

class RaEncoder(nn.Module):
    def __init__(self, spinput_channel, spbase_channel, painput_channel, 
                    out_channel, panum_heads, panum_layers, norm_cfg=dict(type='GN', requires_grad=True)):
        super().__init__()
        self.record_time = False
        block = post_act_block
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(spinput_channel, spbase_channel, 1),
            nn.GroupNorm(8, spbase_channel),
            nn.ReLU(inplace=True))
        self.conv1 = spconv.SparseSequential(
            block(spbase_channel, spbase_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(spbase_channel*2, spbase_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(spbase_channel*2, spbase_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
        )
        self.conv2 = spconv.SparseSequential(
            block(spbase_channel*2, spbase_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(spbase_channel*8, spbase_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(spbase_channel*8, spbase_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(spbase_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        # patch attn
        self.patch_attn = PatchSelfAttention(input_channels=painput_channel, output_channels=out_channel, 
                                             embed_dim=out_channel, num_heads=panum_heads, num_layers=panum_layers)
        # output
        self.out = nn.Sequential(
            nn.Conv3d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=3, padding=1),
            build_norm_layer(norm_cfg, out_channel)[1],
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1))

    def forward(self, ra_sp, ra_pf, ra_pe):
        x = self.conv_out(self.conv2(self.conv1(self.conv_input(ra_sp))))
        ranl, azil, elel = x.spatial_shape
        
        bsr = x.batch_size
        x_dense = x.dense()
        xx = self.patch_attn(ra_pf, ra_pe)
        
        xx_dense = xx.view(bsr, ranl, azil, elel, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = self.out(torch.cat([x_dense, xx_dense], dim=1))
        return out  