import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.NetUtil import *

class OccFlowHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='GN', requires_grad=True, num_groups=24)):
        super(OccFlowHead, self).__init__()
        self.occ_conv = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, in_channels*2)[1],
                nn.ReLU(inplace=True))
        self.occ_pred_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, in_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, in_channels//2)[1],
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, in_channels//4)[1],
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels//4, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        self.flow_pred_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, in_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, in_channels//2)[1],
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, in_channels//4)[1],
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=in_channels//4, out_channels=3, kernel_size=1, stride=1, padding=0))
     
    def forward(self, occ_cf):
        occ_f = self.occ_conv(occ_cf)
        occ_ms = self.occ_pred_conv(occ_f)
        occ_sf = self.flow_pred_conv(occ_f)

        return occ_ms, occ_sf
    
class OccHead(nn.Module):
    def __init__(self, in_channels, out_channels, act_cfg=dict(type='LeakyReLU'), drop=0., bias=False):
        super(OccHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//2, bias=bias)
        self.act = build_act_layer(act_cfg)[1]
        self.fc2 = nn.Linear(in_channels//2, out_channels, bias=bias)
        self.drop1 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop1(x)
        return x
    
class FlowHead(nn.Module):
    def __init__(self, in_channels, act_cfg=dict(type='LeakyReLU'), drop=0., bias=False):
        super(FlowHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//2, bias=bias)
        self.act = build_act_layer(act_cfg)[1]
        self.fc2 = nn.Linear(in_channels//2, 3, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
