import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from Models.OccEncoderUtil import *
from Models.AttentionUtil import TransFormerLayer
from Models.Head import OccHead, FlowHead
from Lib.corr_2d import correlation

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

class OccCorrelation(nn.Module):

    def __init__(self, md=4, occ_planes=[64,128,256,512], num_layers=1, embed_dims=256, num_levels=1, num_points=8, num_heads=4, 
                feedforward_channels=1024, ffn_dropout=0.0, ffn_num_fcs=2, value_dim=3,
                act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')):
        """
        input: md --- maximum 2d displacement (for correlation. default: 4), after warpping
        """
        super(OccCorrelation, self).__init__()
        # init 
        nd = (2*md+1)**2
        cc = [128,96,64,32]
        dd = np.cumsum([128,96,64,32])
        level_channels = []
        ir3dc = [1, 7, 19, 35]
        # corr 
        self.corr = correlation.Correlation2D(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # decoder
        occ_planes = occ_planes[::-1]
        self.dense_convs = nn.ModuleList()
        self.input_proj = nn.ModuleList()
        self.deform_attn = nn.ModuleList()
        self.mask_head = nn.ModuleList()
        self.flow_head = nn.ModuleList()
        for lvl in range(len(occ_planes)):
            if lvl == 0:
                od = nd
            else:
                od = nd+occ_planes[lvl]+4
            level_convs = nn.ModuleList([
                conv(od, cc[0]),
                conv(od+dd[0], cc[1]),
                conv(od+dd[1], cc[2]),
                conv(od+dd[2], cc[3]),
            ])
            proj = nn.Linear(occ_planes[lvl]+3, embed_dims)
            attn = TransFormerLayer(
                embed_dims=embed_dims, num_levels=num_levels, num_points=num_points, num_heads=num_heads, value_dim=value_dim,
                feedforward_channels=feedforward_channels, ffn_dropout=ffn_dropout, ffn_num_fcs=ffn_num_fcs,
                act_cfg=act_cfg, norm_cfg=norm_cfg)
            ohead = OccHead(in_channels=embed_dims, out_channels=3)
            fhead = FlowHead(in_channels=ir3dc[lvl])
            self.dense_convs.append(level_convs)
            self.input_proj.append(proj)
            self.deform_attn.append(attn)
            self.mask_head.append(ohead)
            self.flow_head.append(fhead)
            level_channels.append(od+dd[3])
        # predict flow
        self.predict_flow = nn.ModuleList([nn.Conv2d(in_channels=channels_l, out_channels=2, kernel_size=3, stride=1, padding=1)
            for channels_l in level_channels])
        # up feat and flow
        self.upfeat = nn.ModuleList([nn.ConvTranspose2d(in_channels=channels_l, out_channels=2, kernel_size=4, stride=2, padding=1)
            for channels_l in level_channels])
        self.upflow = nn.ModuleList([nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
            for channels_l in level_channels])
        self.flow_head[0] = nn.Identity()
        self.upfeat[-1] = nn.Identity()
        self.upflow[-1] = nn.Identity()

        # 3d deformable
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels 
        self.num_points = num_points
        self.value_dim = value_dim
        # self.query_proj = nn.Linear(occ_planes[-1], embed_dims)
        # self.value_proj = nn.Linear(occ_planes[-1], embed_dims)
        # self.layers = nn.ModuleList(
        #     [TransFormerLayer(
        #         embed_dims=embed_dims, num_levels=num_levels, num_points=num_points, num_heads=num_heads, value_dim=value_dim,
        #         feedforward_channels=feedforward_channels, ffn_dropout=ffn_dropout, ffn_num_fcs=ffn_num_fcs,
        #         act_cfg=act_cfg, norm_cfg=norm_cfg) for _ in range(num_layers)]
        # )

        # irregular cost valumn
        offsets0 = torch.tensor([[ 0,  0,  0]])
        offsets1 = torch.tensor([
                    [ 0,  0,  0], [ 0,  0,  1], [ 0,  0, -1], [ 1,  0,  0], [-1,  0,  0],
                    [ 0,  1,  0], [ 0, -1,  0]])
        offsets2 = torch.tensor([
                    [1,1,0],[1,0,0],[1,-1,0], [0,1,0],[0,0,0],[0,-1,0], [-1,1,0],[-1,0,0],[-1,-1,0], # level0
                    [1,0,1], [0,1,1],[0,0,1],[0,-1,1],[-1,0,1], # level1
                    [1,0,-1],[0,1,-1],[0,0,-1],[0,-1,-1],[-1,0,-1] # level-1
                    ])
        offsets3 = torch.tensor([
                    [2,1,0],[2,0,0],[2,-1,0],[1,1,0],[1,0,0],[1,-1,0], 
                    [0,1,0],[0,0,0],[0,-1,0], [-1,1,0],[-1,0,0],[-1,-1,0],[-2,1,0],[-2,0,0],[-2,-1,0], # level0
                    [1,1,1],[1,0,1],[1,-1,1], [0,1,1],[0,0,1],[0,-1,1], [-1,1,1],[-1,0,1],[-1,-1,1], # level1
                    [1,1,-1],[1,0,-1],[1,-1,-1], [0,1,-1],[0,0,-1],[0,-1,-1], [-1,1,-1],[-1,0,-1],[-1,-1,-1], # level-1
                    [0,0,2],[0,0,-2]])
        self.register_buffer('offsets0', offsets0)
        self.register_buffer('offsets1', offsets1)
        self.register_buffer('offsets2', offsets2)
        self.register_buffer('offsets3', offsets3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
    def flow_to_ref(self, flow3d, shape):
        # init
        bl, dl, hl, wl = shape
        device = flow3d.device
        dtype = flow3d.dtype
        # flatten
        flow3d = flow3d.view(bl, 3, -1).permute(0,2,1).contiguous() # [B, N, 3]
        flow3d[:, :, 0] = flow3d[:, :, 0]/dl
        flow3d[:, :, 1] = flow3d[:, :, 1]/hl
        flow3d[:, :, 2] = flow3d[:, :, 2]/wl
        # grid 
        grid_x = (torch.arange(dl, device=device) + 0.5) / dl
        grid_y = (torch.arange(hl, device=device) + 0.5) / hl
        grid_z = (torch.arange(wl, device=device) + 0.5) / wl
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=-1)  # [D, H, W, 3]
        grid = grid.view(-1, 3)  # [N, 3]
        point = grid[None,:,:] + flow3d
        point = point[:, :, [2, 1, 0]]
        return point
    
    def grid_sample_select(self, feat2, flow2d_select, idx, offsets):
        bl, cl, dl, hl, wl = feat2.shape
        device = feat2.device
        dtype = feat2.dtype
        nl = flow2d_select.shape[2]                                                          # flow2d_select = shape[B, 2, N]
        # grid
        grid_x = (torch.arange(dl, device=device) + 0.5) / dl
        grid_y = (torch.arange(hl, device=device) + 0.5) / hl
        grid_z = (torch.arange(wl, device=device) + 0.5) / wl
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=-1)  # [D, H, W, 3]
        indices_expanded = idx.unsqueeze(1).expand(-1, 3, -1)
        grid_select = grid.view(-1, 3).unsqueeze(0).expand(bl, -1, -1).permute(0,2,1).contiguous()
        grid_select = torch.gather(grid_select, dim=2, index=indices_expanded)                # B 3 N
        grid_select[:, 0] += flow2d_select[:, 0] / (dl-1) * 2 - 1 
        grid_select[:, 1] += flow2d_select[:, 1] / (dl-1) * 2 - 1 
        grid_select = grid_select.permute(0,2,1).contiguous()
        # normalize
        num_cost = offsets.shape[0]
        offsets[:, 0] *= 2.0 / (dl-1); offsets[:, 1] *= 2.0 / (hl-1); offsets[:, 2] *= 2.0 / (wl - 1)   # d -> x # h -> y # w -> z
        grid_select = grid_select.unsqueeze(2) + offsets.view(1, 1, num_cost, 3)  # [B, N, 7, 3]
        grid_select = grid_select[:, :, :, [2, 1, 0]].view(bl, nl, num_cost, 1, 3)
        feat2_sampled = F.grid_sample(feat2, grid_select, mode='bilinear',align_corners=True).squeeze()     # [B, C, N, 7]
        return feat2_sampled
           
    
    def get_ref_3d(self, x, shape):
        # init
        bl, dl, hl, wl = shape
        device = x.device
        dtype = x.dtype
       # grid 
        grid_x = (torch.arange(dl, device=device) + 0.5) / dl
        grid_y = (torch.arange(hl, device=device) + 0.5) / hl
        grid_z = (torch.arange(wl, device=device) + 0.5) / wl
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=-1)  # [D, H, W, 3]
        grid = grid.view(-1, 3)  # [N, 3]
        point = grid.unsqueeze(0).expand(bl, -1, -1)
        point = point[:, :, [2, 1, 0]].to(dtype=dtype)
        return point

    def forward(self, occf1, occf2, pos_corr):
        # init 
        flow2d = None
        flow2dup = None
        feat2dup = None
        flows2d = []
        flows3d = []
        masks3d = []
        offsets = []
        occf1 = occf1[::-1]; occf2 = occf2[::-1]
        bso, co, do, ho, wo = occf1[-1].shape
        pos_corr_spatial = pos_corr.transpose(1, 2).view(bso, -1, do, ho, wo).contiguous()
        offsets.append(self.offsets0.clone().float()); offsets.append(self.offsets1.clone().float())
        offsets.append(self.offsets2.clone().float()); offsets.append(self.offsets3.clone().float())
        # recurrent
        for lvl, (occf1_l, occf2_l) in enumerate(zip(occf1, occf2)):
            bsl, cl, dl, hl, wl = occf1_l.shape
            dtype = occf1_l.dtype
            device = occf1_l.device
            # first layer
            if lvl == 0:
                occf1_l2d0 = occf1_l[:, :, :, :, 0]; occf1_l2d1 = occf1_l[:, :, :, :, 1]; 
                occf2_l2d0 = occf2_l[:, :, :, :, 0]; occf2_l2d1 = occf2_l[:, :, :, :, 1]; 
                corr0 = self.corr(occf1_l2d0, occf2_l2d0); corr1 = self.corr(occf1_l2d1, occf2_l2d1)
                corr0 = self.leakyRELU(corr0); corr1 = self.leakyRELU(corr1)
                for conv in self.dense_convs[lvl]:
                    corr0 = torch.cat([conv(corr0), corr0], dim=1)
                    corr1 = torch.cat([conv(corr1), corr1], dim=1)
                flow2d0 = self.predict_flow[lvl](corr0); flow2d1 = self.predict_flow[lvl](corr1)
                flow2d = torch.stack([flow2d0, flow2d1], dim=-1)
                occ_sf_l = torch.cat([flow2d, torch.zeros(bsl, 1, dl, hl, wl, dtype=dtype, device=device)], dim=1)
                y = torch.cat([occf1_l, occ_sf_l], dim=1); y = y.view(bsl, cl+3, dl*hl*wl).permute(0, 2, 1).contiguous()
                y = self.input_proj[lvl](y)
                pos = F.avg_pool3d(pos_corr_spatial, kernel_size=2**(3-lvl), stride=2**(3-lvl))
                occ_ms_l = self.deform_attn[lvl](query=y, value=y, query_pos=pos.flatten(2).transpose(1, 2).contiguous(),
                           ref=self.get_ref_3d(y, [bsl, dl, hl, wl]).unsqueeze(2).expand(-1, -1, self.num_levels, -1),
                           spatial_shapes=torch.tensor([[dl, hl, wl]], dtype=torch.long, device=device), 
                           level_start_index=torch.tensor([0], dtype=torch.long, device=device))
                occ_ms_l = self.mask_head[lvl](occ_ms_l)
                occ_ms_l = occ_ms_l.view(bsl, dl, hl, wl, -1).permute(0, 4, 1, 2, 3).contiguous()
                flows2d.append(flow2d0 * 0.5 + flow2d1 * 0.5)
                flow2dup = self.upflow[lvl](flow2d0 * 0.5 + flow2d1 * 0.5)
                feat2dup = self.upfeat[lvl](corr0 * 0.5 + corr1 * 0.5)
                flows3d.append(occ_sf_l)
                masks3d.append(occ_ms_l)
                continue
            # 3d to 2d
            occ_prob = F.interpolate(occ_ms_l, scale_factor=2, mode='trilinear', align_corners=False).softmax(dim=1)
            weight_h = 0.00001 * occ_prob[:, 0] + 0.25 * occ_prob[:, 1] + 0.75 * occ_prob[:, 2]
            flat_weight = weight_h.view(bsl, -1)
            _, topk_indices = torch.topk(flat_weight, k=(dl*hl*wl)//10, dim=1)
            occf1_l2d = torch.logsumexp(occf1_l+torch.log(weight_h.unsqueeze(1) + 1e-6), dim=4)
            occf2_l2d = torch.logsumexp(occf2_l+torch.log(weight_h.unsqueeze(1) + 1e-6), dim=4)
            # warp
            occf2_l2d = self.warp(occf2_l2d, flow2dup*2.0)
            # corr
            corr = self.corr(occf1_l2d, occf2_l2d)
            corr = self.leakyRELU(corr)
            # feat decoder
            x = torch.cat([corr, occf1_l2d, flow2dup, feat2dup], dim=1)
            for conv in self.dense_convs[lvl]:
                x = torch.cat([conv(x), x], dim=1)
            # predict flow
            flow2d = self.predict_flow[lvl](x)
            flows2d.append(flow2d)
            # up
            flow2dup = self.upflow[lvl](flow2d)
            feat2dup = self.upfeat[lvl](x)
            # 2d to 3d
            # occf1_select
            occf1_l_select = occf1_l.view(bsl, cl, -1)
            indices_expanded = topk_indices.unsqueeze(1).expand(-1, cl, -1)
            occf1_l_select = torch.gather(occf1_l_select, dim=2, index=indices_expanded)
            # flow2d select
            flow2d_select = flow2d.unsqueeze(-1).expand(-1, -1, -1, -1, wl)
            flow2d_select = flow2d_select.reshape(bsl, 2, -1)
            indices_expanded = topk_indices.unsqueeze(1).expand(-1, 2, -1)
            flow2d_select = torch.gather(flow2d_select, dim=2, index=indices_expanded)  # [B, 2, N]
            occf2_l_select = self.grid_sample_select(occf2_l, flow2d_select, topk_indices, offsets[lvl]) # [B, C, N, num_offset]
            cost3d = torch.sum(occf1_l_select.unsqueeze(-1) * occf2_l_select, dim=1)   # [B, N, num_offset]
            flow3d = self.flow_head[lvl](cost3d)
            flow3d = flow3d.permute(0,2,1).contiguous()
            flow3d[:, :2, :] += flow2d_select
            occ_sf_l = torch.zeros(bsl, 3, dl*hl*wl, dtype=flow3d.dtype, device=flow3d.device)
            indices_expanded = topk_indices.unsqueeze(1).expand(-1, 3, -1)
            occ_sf_l.scatter_(dim=2, index=indices_expanded, src=flow3d)
            occ_sf_l = occ_sf_l.view(bsl, 3, dl, hl, wl)
            y = torch.cat([occf1_l, occ_sf_l], dim=1); y = y.view(bsl, cl+3, dl*hl*wl).permute(0, 2, 1).contiguous()
            y = self.input_proj[lvl](y)
            pos = F.avg_pool3d(pos_corr_spatial, kernel_size=2**(3-lvl), stride=2**(3-lvl))
            occ_mp_f = self.deform_attn[lvl](query=y, value=y, query_pos=pos.flatten(2).transpose(1, 2).contiguous(), 
                           ref=self.get_ref_3d(y, [bsl, dl, hl, wl]).unsqueeze(2).expand(-1, -1, self.num_levels, -1),
                           spatial_shapes=torch.tensor([[dl, hl, wl]], dtype=torch.long, device=device), 
                           level_start_index=torch.tensor([0], dtype=torch.long, device=device))  # B N C
            occ_ms_l = self.mask_head[lvl](occ_mp_f)
            occ_ms_l = occ_ms_l.view(bsl, dl, hl, wl, -1).permute(0, 4, 1, 2, 3).contiguous()
            flows3d.append(occ_sf_l)
            masks3d.append(occ_ms_l)
        # '''
        # deformable 
        # '''
        # # pre Query
        # query = occf1[-1]
        # device = query.device
        # bso, co, dl, hl, wl = query.shape
        # query = query.view(bso, co, dl*hl*wl).permute(0, 2, 1).contiguous()
        # query = self.query_proj(query)
        # # pre Value
        # value = occf2[-1]
        # value = value.view(bso, co, dl*hl*wl).permute(0, 2, 1).contiguous()
        # value = self.value_proj(value)
        # spatial_shapes = [[dl, hl, wl]]
        # value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device) # N * 3
        # level_start_index = [0]
        # value_level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=device) # N * 1
        # # pre Ref3d
        # ref_3d_on_flow = self.flow_to_ref(occ_sf_l, [bso, dl, hl, wl])
        # ref_3d_on_flow = ref_3d_on_flow.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        # # deform attn
        # occ_cf = query
        # for layer in self.layers:  # self.layers 是 ModuleList
        #     occ_cf = layer(query=occ_cf, value=value, query_pos=pos_corr,ref=ref_3d_on_flow,
        #                    spatial_shapes=value_spatial_shapes, level_start_index=value_level_start_index)
        occ_mp_f = occ_mp_f.view(bso, do, ho, wo, -1).permute(0, 4, 1, 2, 3).contiguous()

        return occ_mp_f, flows2d, flows3d, masks3d

        
