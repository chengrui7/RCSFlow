import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

from Models.NetUtil import *

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_cfg=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(inplace=True),
    )

    return m

class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, norm_cfg=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        padding = (kernel_size-1)//2

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out
    

class PatchSelfAttention(nn.Module):
    def __init__(self, input_channels, output_channels, embed_dim, num_heads, num_layers):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.0,
            activation="relu",
            batch_first=True, 
        )
        self.input_proj = nn.Linear(input_channels, embed_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(embed_dim, output_channels)
    
    def forward(self, x, position_embedding):
        # x: [B, N, C]
        x = self.input_proj(x)
        x = self.encoder(x+position_embedding)
        x = self.output_proj(x)
        return x
