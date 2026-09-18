import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MotionSegLoss(nn.Module):
    def __init__(self, params):
        super(MotionSegLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32))
        
    def forward(self, pred, label):
        loss = self.loss(pred, label)
        return loss

class SceneFlowLoss(nn.Module):
    def __init__(self, params):
        super(SceneFlowLoss, self).__init__()
        self.occ_voxel = params['model']['occ']['voxel']
        self.loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, pred, labelsf, labelmask):
        labelmask = labelmask.unsqueeze(1).float()
        pred = pred * self.occ_voxel
        loss = self.loss(pred, labelsf)
        loss = loss*labelmask
        loss = loss.sum() / (labelmask.sum() + 1e-12)
        return loss

class SegFlowConsistencyLoss(nn.Module):
    def __init__(self, params):
        super(SegFlowConsistencyLoss, self).__init__()
        self.occ_voxel = params['model']['occ']['voxel']
        self.loss = nn.SmoothL1Loss(reduction='none')
        self.closs = nn.CrossEntropyLoss(torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32))

    def forward(self, pred2ds, pred3ds, mask_pred3ds, labelsf, labelmask):
        # init
        n_pred = len(pred2ds)
        pred2ds = pred2ds[::-1]
        pred3ds = pred3ds[::-1]
        mask_pred3ds = mask_pred3ds[::-1]
        total_loss_f2d = 0
        total_loss_f3d = 0
        total_loss_s3d = 0
        labelmask = labelmask.unsqueeze(1)  # [B,1,H,W,D]
        # label 3d to 2d
        labelsf_2d = labelsf[:, :2, :, :, :] * labelmask  # [B,2,H,W,D]
        valid_count = labelmask.sum(dim=-1)          # [B,1,H,W]
        labelsf_2d = labelsf_2d.sum(dim=-1) / valid_count.clamp(min=1)
        labelmask_2d = labelmask.any(dim=-1)
        # recurrent
        for i in range(n_pred):
            
            pred3d = pred3ds[i]*self.occ_voxel*(2**i)
            pred2d = pred2ds[i]*self.occ_voxel*(2**i)
            labelmask_3d_down = F.max_pool3d(labelmask.float(), kernel_size=2**i, stride=2**i)
            labelmask_3d_down_long = labelmask_3d_down.squeeze(1).long()
            labelsf_3d_down = F.max_pool3d(labelsf, kernel_size=2**i, stride=2**i)
            labelsf_2d_down = F.max_pool2d(labelsf_2d, kernel_size=2**i, stride=2**i)
            labelmask_2d_down = -F.max_pool2d(-labelmask_2d.float(), kernel_size=2**i, stride=2**i)
            loss2df = self.loss(pred2d, labelsf_2d_down)
            loss2df = loss2df * labelmask_2d_down
            loss2df = loss2df.sum() / (labelmask_2d_down.sum() + 1e-12)
            loss3df = self.loss(pred3d, labelsf_3d_down)
            loss3df = loss3df*labelmask_3d_down
            loss3df = loss3df.sum() / (labelmask_3d_down.sum() + 1e-12)
            weight = 0.5 ** i
            total_loss_f2d += weight * loss2df
            total_loss_f3d += weight * loss3df
            total_loss_s3d += weight * self.closs(mask_pred3ds[i], labelmask_3d_down_long)

        return total_loss_f2d, total_loss_f3d, total_loss_s3d

class MotionSegIou():   
    def __init__(self, params):
        super(MotionSegIou, self).__init__()
        num_classes = params['model']['head']['num_cls']
        self.num_classes = num_classes
        self.tp = torch.zeros(num_classes).to('cuda')
        self.fp = torch.zeros(num_classes).to('cuda')
        self.fn = torch.zeros(num_classes).to('cuda')

    def cal_batch(self, pred, label):
        pred = torch.argmax(pred, dim=1)
        # target Iou
        pred_fg = (pred == 1) | (pred == 2)
        label_fg = (label == 1) | (label == 2)
        self.tp[0] += (pred_fg & label_fg).sum()
        self.fp[0] += (pred_fg & ~label_fg).sum()
        self.fn[0] += (~pred_fg & label_fg).sum()
        # static
        self.tp[1] += ((pred == 1) & (label == 1)).sum()
        self.fp[1] += ((pred == 1) & (label != 1)).sum()
        self.fn[1] += ((pred != 1) & (label == 1)).sum()
        # moving
        self.tp[2] += ((pred == 2) & (label == 2)).sum()
        self.fp[2] += ((pred == 2) & (label != 2)).sum()
        self.fn[2] += ((pred != 2) & (label == 2)).sum()

        iou = (pred_fg & label_fg).sum() / ((pred_fg & label_fg).sum() + (pred_fg & ~label_fg).sum() + (~pred_fg & label_fg).sum() + 1e-6)
        iou_static = ((pred == 1) & (label == 1)).sum() / (((pred == 1) & (label == 1)).sum() + ((pred == 1) & (label != 1)).sum() + ((pred != 1) & (label == 1)).sum() + 1e-6)
        iou_moving = ((pred == 2) & (label == 2)).sum() / (((pred == 2) & (label == 2)).sum() + ((pred == 2) & (label != 2)).sum() + ((pred != 2) & (label == 2)).sum() + 1e-6)
        m_iou = (iou_static + iou_moving) / 2.0
        return iou, iou_static, iou_moving, m_iou

    def cal_end(self):
        iou = self.tp[0] / (self.tp[0] + self.fp[0] + self.fn[0] + 1e-6)
        iou_static = self.tp[1] / (self.tp[1] + self.fp[1] + self.fn[1] + 1e-6)
        iou_moving = self.tp[2] / (self.tp[2] + self.fp[2] + self.fn[2] + 1e-6)
        m_iou = (iou_static + iou_moving) / 2.0
        return iou, iou_static, iou_moving, m_iou
    
    def reset(self):
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()

class SceneFlowEPE():   
    def __init__(self, params):
        super(SceneFlowEPE, self).__init__()
        # 3d
        self.epe3d_sum = 0.0
        self.accs3d_sum = 0
        self.accr3d_sum = 0
        self.out3d_sum = 0
        # 2d
        self.epe2d_sum = 0.0
        self.accs2d_sum = 0
        self.accr2d_sum = 0
        self.out2d_sum = 0
        # num
        self.num_sum = 0
        self.occ_voxel = params['model']['occ']['voxel']

    def cal_batch(self, pred_sf, label_sf, pred, label):
        # init
        pred = torch.argmax(pred, dim=1)
        flowmask = ((pred == 1) | (pred == 2)) & ((label == 1) | (label == 2))
        self.num_sum += flowmask.sum()
        pred_sf = pred_sf * self.occ_voxel
        flow_pred_masked = pred_sf.permute(0,2,3,4,1)[flowmask]  # shape [N, 3]
        flow_gt_masked   = label_sf.permute(0,2,3,4,1)[flowmask]    # shape [N, 3]
        gt_norm = torch.norm(flow_gt_masked, dim=1)  # [N]
        gt_norm2d = torch.norm(flow_gt_masked[:, :2], dim=1)
        # endpoint error
        epe3d = torch.norm(flow_pred_masked - flow_gt_masked, dim=1)
        rel_err = epe3d / (gt_norm + 1e-6)
        accs3d_mask = (epe3d < 0.05) | (rel_err < 0.05)
        accr3d_mask = (epe3d < 0.1) | (rel_err < 0.1)
        out3d_mask = (epe3d > 0.3) | (rel_err > 0.1)
        self.epe3d_sum += epe3d.sum()
        self.accs3d_sum += accs3d_mask.float().sum()
        self.accr3d_sum += accr3d_mask.float().sum()
        self.out3d_sum += out3d_mask.float().sum()
        epe2d = torch.norm(flow_pred_masked[:, :2] - flow_gt_masked[:, :2], dim=1)
        rel_err2d = epe2d / (gt_norm2d + 1e-6)
        accs2d_mask = (epe2d < 0.05) | (rel_err2d < 0.05)
        accr2d_mask = (epe2d < 0.1) | (rel_err2d < 0.1)
        out2d_mask = (epe2d > 0.3) | (rel_err2d > 0.1)
        self.epe2d_sum += epe2d.sum()
        self.accs2d_sum += accs2d_mask.float().sum()
        self.accr2d_sum += accr2d_mask.float().sum()
        self.out2d_sum += out2d_mask.float().sum()
        return epe3d.sum()/flowmask.sum(), accs3d_mask.float().sum()/flowmask.sum(), accr3d_mask.float().sum()/flowmask.sum(), out3d_mask.float().sum()/flowmask.sum(), \
               epe2d.sum()/flowmask.sum(), accs2d_mask.float().sum()/flowmask.sum(), accr2d_mask.float().sum()/flowmask.sum(), out2d_mask.float().sum()/flowmask.sum()
        
    def cal_end(self):
        mepe3d = self.epe3d_sum / self.num_sum
        accs3d = self.accs3d_sum / self.num_sum
        accr3d = self.accr3d_sum / self.num_sum
        out3d = self.out3d_sum / self.num_sum
        # 2d
        mepe2d = self.epe2d_sum / self.num_sum
        accs2d = self.accs2d_sum / self.num_sum
        accr2d = self.accr2d_sum / self.num_sum
        out2d = self.out2d_sum / self.num_sum
        return mepe3d, accs3d, accr3d, out3d, mepe2d, accs2d, accr2d, out2d
    
    def reset(self):
        self.epe3d_sum = 0.0
        self.accs3d_sum = 0
        self.accr3d_sum = 0
        self.out3d_sum = 0
        # 2d
        self.epe2d_sum = 0.0
        self.accs2d_sum = 0
        self.accr2d_sum = 0
        self.out2d_sum = 0
        # num
        self.num_sum = 0


