import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.NetUtil import *
from Models.AttentionUtil import TransFormerLayer

class RaImDeformAttn(nn.Module):
    def __init__(self, params, num_layers, embed_dims=256, num_levels=1, num_points=8, num_heads=4,
                feedforward_channels=1024, ffn_dropout=0.0, ffn_num_fcs=2, value_dim=2,
                act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')):
        super(RaImDeformAttn, self).__init__()
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
        # ex/intrinsics
        self.register_buffer('intrinsics', params['intrinsics'].clone())
        self.register_buffer('imgsize', params['imgsize'].clone())
        self.RadartoImage = Extrinsics(params['extrinsicsRadar2Cam'].clone())
        # attn
        radar_reference_pixels = self.radar_to_image()
        self.register_buffer('radar_reference_pixels', radar_reference_pixels.clone())
        self.layers = nn.ModuleList([
            TransFormerLayer(
                embed_dims=embed_dims, num_levels=num_levels, num_points=num_points, value_dim=value_dim, num_heads=num_heads, 
                feedforward_channels=feedforward_channels, ffn_dropout=ffn_dropout, ffn_num_fcs=ffn_num_fcs,
                act_cfg=act_cfg, norm_cfg=norm_cfg) for _ in range(num_layers)
        ])
        self.deconv1 = DeconvBlock3D(embed_dims, embed_dims//2)
        self.deconv2 = DeconvBlock3D(embed_dims//2, embed_dims//4)

    def radar_to_image(self):
        def downsample_coords(arr, factor):
            N = arr.shape[0]
            assert N % factor == 0
            arr = arr.view(N // factor, factor)
            return arr.mean(dim=1)
        # init
        ran, azi, ele = self.r_dim//4, self.a_dim//4, self.e_dim//4
        # radar polar
        arrR_new = downsample_coords(self.arrR, self.r_dim//ran)
        arrA_new = downsample_coords(self.arrA, self.a_dim//azi)
        arrE_new = downsample_coords(self.arrE, self.e_dim//ele)
        r_coords = arrR_new.view(ran, 1, 1).expand(ran, azi, ele)
        a_coords = arrA_new.view(1, azi, 1).expand(ran, azi, ele)
        e_coords = arrE_new.view(1, 1, ele).expand(ran, azi, ele)
        coords = torch.stack([r_coords, a_coords, e_coords], dim=0) # 3 R A E
        coords = coords.unsqueeze(0)
        coords = coords.to(dtype=torch.float32)
        coords = coords.view(1, 3, -1)  # [1,3,rae]
        # radar coords
        points = spherical_to_cartesian(coords)
        # radar to image
        cam_points = self.RadartoImage.R.unsqueeze(0) @ points + self.RadartoImage.t.unsqueeze(0)
        cam_points = cam_points / (cam_points[:, 2:3, :] + 1e-6)      # normlize 1 3 N
        intrinsics = self.intrinsics.unsqueeze(0).repeat(1, 1, 1)
        radaruv = torch.bmm(intrinsics, cam_points)          
        radaruv = radaruv[:, :2, :]                                   # 1 2 N
        radar_reference_pixels = radaruv / self.imgsize.view(1,2,1)
        #radar_reference_pixels = radar_reference_pixels[:, [1,0], :]  # 1 2 N (v_grid, u_grid)
        radar_reference_pixels = radar_reference_pixels.permute(0,2,1).contiguous()  # 1 N 2 (u_grid, v_grid)
        # # check collision
        # radaruv_px = torch.floor(radaruv).long()
        # pixel_id = radaruv_px[:, 1, :] * self.imgsize[0] + radaruv_px[:, 0, :]  # 1 N
        # unique_ids, counts = torch.unique(pixel_id[0], return_counts=True)
        # collision_mask = counts > 1                 # K (num K for unique_ids)
        # collision_pixel_ids = unique_ids[collision_mask]    # M (num M collision_pixel_ids in unique_ids)
        # collision_dict = {}
        # for pid in collision_pixel_ids:
        #     voxel_idx = torch.nonzero(pixel_id[0] == pid, as_tuple=False).squeeze(1)
        #     e_index = voxel_idx % ele
        #     a_index = (voxel_idx // ele) % azi
        #     r_index = voxel_idx // (azi * ele)
        #     u_image = int(pid.item()) % self.imgsize[0]
        #     v_image = int(pid.item()) // self.imgsize[0]
        #     collision_dict[[v_image, u_image]] = [r_index, a_index, e_index]
            
        return radar_reference_pixels

    def forward(self, raf, imf, pos):
        # init
        device = imf[0].device
        # pre Query and Key
        bsr, cr, ran, azi, ele = raf.shape
        query = raf.view(bsr, cr, ran*azi*ele).permute(0, 2, 1).contiguous()
        key = torch.zeros_like(query)
        # pre Value
        value_flatten = []
        spatial_shapes = []
        grid_2d_on_image = (self.radar_reference_pixels.expand(bsr, -1, -1).view(bsr, ran*azi*ele, 1, 2)) * 2 - 1  #[-1,1]
        for iml in imf:
            bsi, cvalue, hl, wl = iml.shape
            spatial_shapes.append([hl, wl])
            feat_sampled = F.grid_sample(iml, grid_2d_on_image, mode='bilinear',align_corners=False).squeeze(-1).permute(0,2,1).contiguous()
            key += query * feat_sampled
            # flatten: [B, C, D*H*W] → permute to [B, D*H*W, C]
            iml = iml.view(bsi, cvalue, hl*wl).permute(0, 2, 1).contiguous()
            value_flatten.append(iml)
        value = torch.cat(value_flatten, dim=1).contiguous()    
        value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device) # N * 3
        level_start_index = [0]
        for shape in spatial_shapes:
            h, w = shape
            level_start_index.append(level_start_index[-1] + h * w)
        value_level_start_index = torch.tensor(level_start_index[:-1], dtype=torch.long, device=device) # N * 1
        # pre Ref2d
        ref_2d_on_image = self.radar_reference_pixels.expand(bsr, -1, -1)  # B N 2
        ref_2d_on_image = ref_2d_on_image.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        # deform attn
        x = query
        for layer in self.layers:  # self.layers 是 ModuleList
            x = layer(query=x, key=key, value=value, query_pos=pos, ref=ref_2d_on_image,
                      spatial_shapes=value_spatial_shapes, level_start_index=value_level_start_index)
        query_enhance_by_image = x.view(bsr, ran, azi, ele, -1).permute(0, 4, 1, 2, 3).contiguous()
        query_enhance_by_image = self.deconv2(self.deconv1(query_enhance_by_image))
        
        return query_enhance_by_image
