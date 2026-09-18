import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Losses.RCOFLossUtil import *

class RCOFLoss(nn.Module):
    def __init__(self, params):
        super(RCOFLoss, self).__init__()
        self.params = params
        self.sfl = SceneFlowLoss(params)
        self.msl = MotionSegLoss(params)
        self.sfcl = SegFlowConsistencyLoss(params)

    def forward(self, occ_ms, occ_sf, occ_intern, occ_mov, occ_sfl, occ_mask):

        # supervise
        occ_sf2d_intern, occ_sf3d_intern, occ_ms3d_intern = occ_intern
        msl_ = self.msl(occ_ms, occ_mov)
        sfl_ = self.sfl(occ_sf, occ_sfl, occ_mask)
        fcl2_, fcl3_, scl_ = self.sfcl(occ_sf2d_intern, occ_sf3d_intern, occ_ms3d_intern, occ_sfl, occ_mask)

        total_loss = msl_*0.4 + 0.05*scl_ + sfl_*0.53 + 0.01*fcl2_ + 0.01*fcl3_
        loss_item = {
            'totalLoss': total_loss,
            'motionsegLoss': msl_,
            'sceneflowLoss': sfl_,
            'segconsistencyLoss': scl_,
            'flow3dconsistencyLoss': fcl3_,
            'flow2dconsistencyLoss': fcl2_,
        }
        # print(loss_item)
        return total_loss, loss_item
    
class RCOFValLoss(nn.Module):
    def __init__(self, params):
        super(RCOFValLoss, self).__init__()
        self.params = params
        self.sfl = SceneFlowLoss(params)
        self.msl = MotionSegLoss(params)

    def forward(self, occ_ms, occ_sf, occ_mov, occ_sfl, occ_mask):

        # supervise
        msl_ = self.msl(occ_ms, occ_mov)
        sfl_ = self.sfl(occ_sf, occ_sfl, occ_mask)

        total_loss = msl_*0.3 + sfl_*0.7 
        loss_item = {
            'totalLoss': total_loss,
            'motionsegLoss': msl_,
            'sceneflowLoss': sfl_,
        }
        return total_loss, loss_item
    
class RCOFTest():
    def __init__(self, params):
        super(RCOFTest, self).__init__()
        self.params = params
        self.sf_cal = SceneFlowEPE(params)
        self.ms_cal = MotionSegIou(params)

    def cal_batch(self, occ_ms, occ_sf, occ_mov, occ_sfl, occ_mask):
        iou, iou_static, iou_moving, m_iou = self.ms_cal.cal_batch(occ_ms, occ_mov)
        mepe3d, accs3d, accr3d, out3d, mepe2d, accs2d, accr2d, out2d = self.sf_cal.cal_batch(occ_sf, occ_sfl, occ_ms, occ_mov)
        metrics_dict = {
            # motion segmentation
            "ioufg": iou.item(),
            "ious": iou_static.item(),
            "ioum": iou_moving.item(),
            "mIoU": m_iou.item(),

            # scene flow 3D
            "epe3d": mepe3d.item(),
            "AccS3d": accs3d.item(),
            "AccR3d": accr3d.item(),
            "Outlier3d": out3d.item(),

            # scene flow 2D
            "epe2d": mepe2d.item(),
            "AccS2d": accs2d.item(),
            "AccR2d": accr2d.item(),
            "Outlier2d": out2d.item()
        }
        return metrics_dict

    def cal_end(self):
        iou, iou_static, iou_moving, m_iou = self.ms_cal.cal_end()
        mepe3d, accs3d, accr3d, out3d, mepe2d, accs2d, accr2d, out2d = self.sf_cal.cal_end()
        metrics_dict = {
            # motion segmentation
            "iou/fg": iou.item(),
            "iou/static": iou_static.item(),
            "iou/moving": iou_moving.item(),
            "iou/mIoU": m_iou.item(),

            # scene flow 3D
            "sf3d/EPE": mepe3d.item(),
            "sf3d/AccS": accs3d.item(),
            "sf3d/AccR": accr3d.item(),
            "sf3d/Outlier": out3d.item(),

            # scene flow 2D
            "sf2d/EPE": mepe2d.item(),
            "sf2d/AccS": accs2d.item(),
            "sf2d/AccR": accr2d.item(),
            "sf2d/Outlier": out2d.item()
        }
        return metrics_dict
    
    def reset(self):
        self.ms_cal.reset()
        self.sf_cal.reset()

    
    
