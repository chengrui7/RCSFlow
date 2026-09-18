import torch
import torch.nn as nn
import torch.nn.functional as F

class DeconvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm_cfg=None):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch,kernel_size=2,stride=2,bias=False)
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(out_ch, out_ch, kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        return self.conv(self.relu(self.bn(self.deconv(x))))

class Extrinsics(nn.Module):
    def __init__(self, T: torch.Tensor):
        super().__init__()
        self.register_buffer('R', T.clone()[:3, :3])
        self.register_buffer('t', T.clone()[:3, 3:4])
    
    def forward(self, x):
        if x.shape[-1] == 3:
            x_h = torch.cat([x, torch.ones(*x.shape[:-1], 1, device=x.device)], dim=-1)  # 转齐次
        else:
            x_h = x
        
        T_full = torch.eye(4, device=self.R.device, dtype=self.R.dtype)
        T_full[:3, :3] = self.R
        T_full[:3, 3:4] = self.t
        x_transformed = torch.matmul(x_h, T_full.T)
        return x_transformed[..., :3]

def build_norm_layer(cfg, num_features):
    if cfg is None:
        cfg = {}
    layer_type = cfg.get('type', 'GN')
    requires_grad = cfg.get('requires_grad', True)
    if layer_type == 'BN3d':
        norm_layer = nn.BatchNorm3d(num_features)
    elif layer_type == 'GN':
        num_groups = cfg.get('num_groups', 32)
        norm_layer = nn.GroupNorm(num_groups, num_features)
    elif layer_type =='SyncBN': 
        norm_layer = nn.SyncBatchNorm(num_features)
    elif layer_type == 'LN':
        norm_layer = nn.LayerNorm(num_features)
    elif layer_type == 'BN':
        norm_layer = nn.BatchNorm2d(num_features)
    else:
        raise NotImplementedError(f'Norm layer {layer_type} is not implemented')

    for param in norm_layer.parameters():
        param.requires_grad = requires_grad
    return layer_type, norm_layer

def build_act_layer(act_cfg):
    act_type = act_cfg.get('type', 'ReLU')
    inplace = act_cfg.get('inplace', True)
    requires_grad = act_cfg.get('requires_grad', True)
    if act_type == 'ReLU':
        act_layer = nn.ReLU(inplace=inplace)
    elif act_type == 'LeakyReLU':
        negative_slope = act_cfg.get('negative_slope', 0.01)
        act_layer = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == 'GELU':
        act_layer = nn.GELU()
    elif act_type == 'SiLU':
        act_layer = nn.SiLU()
    else:
        raise ValueError(f'Unsupported activation type: {act_type}')
    
    for param in act_layer.parameters():
        param.requires_grad = requires_grad
    return act_type, act_layer

def cartesian_to_spherical(cartesian):
    # xyz
    x, y, z = cartesian[:, 0, :], cartesian[:, 1, :], cartesian[:, 2, :]  # (B, N)
    # rae
    r = torch.sqrt(x**2 + y**2 + z**2)        # (B, N)
    theta = torch.atan2(y, x)                       # (B, N), azimuth
    phi = torch.asin(z/r) # (B, N), elevation

    spherical = torch.stack([r, theta, phi], dim=1)  # (B, 3, N)
    return spherical
    

def spherical_to_cartesian(spherical):
    # rae
    range, azimuth, elevation = spherical[:, 0, :], spherical[:, 1, :], spherical[:, 2, :]
    # xyz
    x = range * torch.cos(elevation) * torch.cos(azimuth)
    y = range * torch.cos(elevation) * torch.sin(azimuth)
    z = range * torch.sin(elevation)

    cartesian = torch.stack([x, y, z], dim=1)  # (B, 3, N)
    return cartesian