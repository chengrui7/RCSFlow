import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.NetUtil import *
from Models.AttentionUtil import TransFormerLayer


class PolarCartDeformAttn(nn.Module):
    def __init__(self, params, num_layers, embed_dims=256, num_levels=1, num_points=8, num_heads=4,
                feedforward_channels=1024, ffn_dropout=0.0, ffn_num_fcs=2, value_dim=3,
                act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')):
        super(PolarCartDeformAttn, self).__init__()
        # init
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels 
        self.num_points = num_points
        self.value_dim = value_dim
        # model
        arrR, arrA, arrE = params['arrR'], params['arrA'], params['arrE']
        self.register_buffer('arrR', arrR.clone())
        self.register_buffer('arrA', arrA.clone())
        self.register_buffer('arrE', arrE.clone())
        self.r_dim, self.a_dim, self.e_dim = len(arrR), len(arrA), len(arrE)
        self.occ_range = params['model']['occ']['range']
        self.occ_voxel = params['model']['occ']['voxel']
        self.occ_size = params['model']['occ']['size']
        # ex/intrinsics
        self.register_buffer('intrinsics', params['intrinsics'].clone())
        self.register_buffer('imgsize', params['imgsize'].clone())
        self.RadartoImage = Extrinsics(params['extrinsicsRadar2Cam'].clone())
        # attn
        occ_reference_polars = self.cart_to_polar()
        self.register_buffer('occ_reference_polars', occ_reference_polars.clone())
        self.cart_voxel = nn.Embedding(self.occ_size[0]*self.occ_size[1]*self.occ_size[2], embed_dims)
        self.layers = nn.ModuleList([
            TransFormerLayer(
                embed_dims=embed_dims, num_levels=num_levels, num_points=num_points, value_dim=value_dim, num_heads=num_heads,
                feedforward_channels=feedforward_channels, ffn_dropout=ffn_dropout, ffn_num_fcs=ffn_num_fcs,
                act_cfg=act_cfg, norm_cfg=norm_cfg) for _ in range(num_layers)
        ])

    def cart_to_polar(self):
        # init
        x_min, y_min, z_min, x_max, y_max, z_max = self.occ_range
        D, H, W = self.occ_size
        xs = torch.linspace(x_min+self.occ_voxel/2, x_max-self.occ_voxel/2, D, dtype=torch.float32)
        ys = torch.linspace(y_min+self.occ_voxel/2, y_max-self.occ_voxel/2, H, dtype=torch.float32)
        zs = torch.linspace(z_min+self.occ_voxel/2, z_max-self.occ_voxel/2, W, dtype=torch.float32)
        xs, ys, zs = torch.meshgrid(xs, ys, zs, indexing='ij')
        # cart to polar
        ref_3d = torch.stack((xs.flatten(), ys.flatten(), zs.flatten()), dim=0) # (3, d*h*w)
        ref_3d = ref_3d.unsqueeze(0)
        ref_3d_polar = cartesian_to_spherical(ref_3d)
        # polar to uni polar
        r_index = (ref_3d_polar[:, 0, :]-self.arrR.min()) / (self.arrR.max()-self.arrR.min())
        a_index = (ref_3d_polar[:, 1, :]-self.arrA.min()) / (self.arrA.max()-self.arrA.min())
        e_index = (ref_3d_polar[:, 2, :]-self.arrE.min()) / (self.arrE.max()-self.arrE.min())   # B N
        occ_reference_polars = torch.stack([e_index, a_index, r_index],dim=-1)   # bsr N 3 
        return occ_reference_polars

    def forward(self, raf, pos):
        # init
        device = raf.device
        dtype = raf.dtype
        bsr, cr, ran, azi, ele = raf.shape
        # pre Query
        query = self.cart_voxel.weight.to(dtype)
        query = query.unsqueeze(0).expand(bsr, -1, -1)
        # pre Value
        bsr, cr, ran, azi, ele = raf.shape
        value = raf.view(bsr, cr, ran*azi*ele).permute(0, 2, 1).contiguous()
        spatial_shapes = [[ran, azi, ele]]
        value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device) # N * 3
        level_start_index = [0]
        value_level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=device) # N * 1
        # pre Ref3d
        ref_3d_on_polar = self.occ_reference_polars.expand(bsr, -1, -1)
        ref_3d_on_polar = ref_3d_on_polar.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        # deform attn
        occf_cart_from_polar = query
        for layer in self.layers:  # self.layers 是 ModuleList
            occf_cart_from_polar = layer(query=occf_cart_from_polar, value=value, query_pos=pos,ref=ref_3d_on_polar,
                                         spatial_shapes=value_spatial_shapes, level_start_index=value_level_start_index)
        occf_cart_from_polar = occf_cart_from_polar.view(bsr, self.occ_size[0], self.occ_size[1], 
                                                         self.occ_size[2], -1).permute(0, 4, 1, 2, 3).contiguous()

        return occf_cart_from_polar