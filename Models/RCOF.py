import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.RaEncoder import RaEncoder
from Models.ImEncoder import ImEncoder
from Models.RaImDeformAttn import RaImDeformAttn
from Models.PolarCartDeformAttn import PolarCartDeformAttn
from Models.PositionEncoding import LearnedPositionEncoding3D
from Models.OccEncoder import OccResNet3D
from Models.OccCorrelation import OccCorrelation
from Models.Head import OccFlowHead

class RCOF(nn.Module):
    def __init__(self, params):
        super(RCOF, self).__init__()
        self.params = params
        # backbone encoder
        self.re = RaEncoder(spinput_channel=params['model']['radar']['sparse_channel'], 
                            spbase_channel=params['model']['radar']['sparse_inplane'], 
                            painput_channel=params['model']['radar']['patch_channel'],  
                            panum_heads=params['model']['radar']['patch_numheads'], 
                            panum_layers=params['model']['radar']['patch_numlayers'],
                            out_channel=params['model']['rida']['embedding_dims'],)
        
        self.ie = ImEncoder(in_channels=params['model']['image']['input_channel'], 
                            base_planes=params['model']['image']['encoder_inplane'], 
                            fpn_out=params['model']['rida']['embedding_dims'])
        # deformable attn for modal fusion
        self.rida = RaImDeformAttn(params=params, 
                                   num_layers=params['model']['rida']['num_layers'],
                                   embed_dims=params['model']['rida']['embedding_dims'], 
                                   num_levels=params['model']['rida']['num_levels'], 
                                   num_points=params['model']['rida']['num_points'], 
                                   num_heads=params['model']['rida']['num_heads'],
                                   feedforward_channels=params['model']['rida']['feedforward_channels'], 
                                   ffn_dropout=params['model']['rida']['ffn_dropout'], 
                                   ffn_num_fcs=params['model']['rida']['ffn_num_fcs'])
        # deformable attn for coords transform
        self.pcda = PolarCartDeformAttn(params=params,
                                        num_layers=params['model']['rida']['num_layers'],
                                        embed_dims=params['model']['pcda']['embedding_dims'], 
                                        num_levels=params['model']['pcda']['num_levels'], 
                                        num_points=params['model']['pcda']['num_points'], 
                                        num_heads=params['model']['pcda']['num_heads'],
                                        feedforward_channels=params['model']['pcda']['feedforward_channels'], 
                                        ffn_dropout=params['model']['pcda']['ffn_dropout'], 
                                        ffn_num_fcs=params['model']['pcda']['ffn_num_fcs'])
        # occ
        self.oeb = OccResNet3D(depth=params['model']['occ']['encoder_depth'],
                            n_input_channels=params['model']['pcda']['embedding_dims'], 
                            block_inplanes=params['model']['occ']['encoder_inplanes'])
        
        self.ocf = OccCorrelation(md=params['model']['occ']['corr_md'], 
                                  occ_planes=params['model']['occ']['encoder_inplanes'], 
                                  num_layers=params['model']['occ']['num_layers'], 
                                  embed_dims=params['model']['occ']['embedding_dims'], 
                                  num_levels=params['model']['occ']['num_levels'], 
                                  num_points=params['model']['occ']['num_points'], 
                                  num_heads=params['model']['occ']['num_heads'],
                                  feedforward_channels=params['model']['occ']['feedforward_channels'], 
                                  ffn_dropout=params['model']['occ']['ffn_dropout'], 
                                  ffn_num_fcs=params['model']['occ']['ffn_num_fcs'])
        # position encoding
        self.ppe = LearnedPositionEncoding3D(num_feats=params['model']['rida']['embedding_dims']//3, 
                                             depth_num_embed=params['model']['radar']['range_dim']//4, 
                                             height_num_embed=params['model']['radar']['azimuth_dim']//4, 
                                             width_num_embed=params['model']['radar']['elevation_dim']//4)
        self.cpe = LearnedPositionEncoding3D(num_feats=params['model']['pcda']['embedding_dims']//3, 
                                             depth_num_embed=params['model']['occ']['size'][0], 
                                             height_num_embed=params['model']['occ']['size'][1], 
                                             width_num_embed=params['model']['occ']['size'][2])
        # head 
        self.head = OccFlowHead(in_channels=params['model']['occ']['embedding_dims'], 
                                out_channels=params['model']['head']['num_cls'])

    def forward(self, im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf, eval_mode=False):
        """
        ra: B, C, topkAE, A, E  (B, 3+C', topkAE, A, E) 
        im: B, 3, H, W
        """
        # init
        device = im0.device
        dtype = im0.dtype
        im0 = 2 * (im0 / 255.0) - 1.0; im1 = 2 * (im1 / 255.0) - 1.0
        bsi, ci, hl, wl = im0.shape
        bsr, np, crp = ra0_pf.shape
        # encoder
        spherical_pos_self = self.ppe(batch_size=bsr)
        imf0 = self.ie(im0); imf1 = self.ie(im1)
        raf0 = self.re(ra0_sp, ra0_pf, spherical_pos_self); raf1 = self.re(ra1_sp, ra1_pf, spherical_pos_self)
        # deform-attn for modal fusion
        spherical_pos_cross = self.ppe(batch_size=bsr)
        raf0_enhance = self.rida(raf0, imf0, spherical_pos_cross)
        raf1_enhance = self.rida(raf1, imf1, spherical_pos_cross)
        # deform-attn for coords transform
        cartesian_pos_cross = self.cpe(batch_size=bsr)
        raf0_enhance_Cartesian = self.pcda(raf0_enhance, cartesian_pos_cross)
        raf1_enhance_Cartesian = self.pcda(raf1_enhance, cartesian_pos_cross)
        # occ feature encoder
        occ0f = self.oeb(raf0_enhance_Cartesian)
        occ1f = self.oeb(raf1_enhance_Cartesian)
        # occ feature corr
        cartesian_pos_corr = self.cpe(batch_size=bsr)
        occ_cf, occ_sf2d, occ_sf3d, occ_ms3d = self.ocf(occ0f, occ1f, cartesian_pos_corr)
        # head
        occ_ms, occ_sf = self.head(occ_cf)
        if eval_mode:
            return occ_ms, occ_sf

        return occ_ms, occ_sf, [occ_sf2d, occ_sf3d, occ_ms3d]