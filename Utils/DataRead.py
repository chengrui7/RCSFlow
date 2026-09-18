import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import scipy.io
import itertools
import time
import math
import os

import spconv.pytorch as spconv

class RCFlowDataset(Dataset):
    def __init__(self, root, seqs, params, mode='train', transform=None):
        """
        Args:
            dataset_dict (dict): dict, key-value have file path and label path
            transform (callable, optional): data transform and augmentation
        """
        self.params = params
        self.transform = transform
        self.mode = mode
        self.files = []
        for seq in seqs:
            dir = os.path.join(root, seq, 'rcof')
            fs = sorted(
                [f for f in os.listdir(dir) if f.endswith(".pt")],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            self.files += [os.path.join(dir, f) for f in fs]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # rcof_data = {
        #     "im0": img0_tensor,
        #     "im1": img1_tensor,
        #     "radar_sparse_feature0": radar_sparse_feature_save,
        #     "radar_sparse_index0": radar_sparse_index_save,
        #     "radar_patch_feature0": radar_patch_feature_save,
        #     "radar_sparse_feature1": radar_sparse_feature_save2,
        #     "radar_sparse_index1": radar_sparse_index_save2,
        #     "radar_patch_feature1": radar_patch_feature_save2,
        #     "occ_ms": occ_ms, 
        #     "occ_sf": occ_sf, 
        # }

        # load data
        # print(self.files[idx])
        sample = torch.load(self.files[idx], weights_only=False)
        im0 = sample["im0"]; im1 = sample["im1"]; 
        rasf0 = sample["radar_sparse_feature0"]; rasc0 = sample["radar_sparse_index0"]; 
        rapf0 = sample["radar_patch_feature0"]
        rasf0[:, :3] = 10 * torch.log10(rasf0[:, :3]); rapf0[:, :6] = 10 * torch.log10(rapf0[:, :6])
        rasf1 = sample["radar_sparse_feature1"]; rasc1 = sample["radar_sparse_index1"]
        rapf1 = sample["radar_patch_feature1"]     
        rasf1[:, :3] = 10 * torch.log10(rasf1[:, :3]); rapf1[:, :6] = 10 * torch.log10(rapf1[:, :6])      
        occ_ms = sample["occ_ms"]; occ_sf = sample["occ_sf"]
        # init gt
        occ_mov = torch.zeros((128, 128, 16), dtype=torch.int64)
        write_val = torch.where(occ_ms[:, 4] == 0, torch.tensor(1).to(torch.int64), torch.tensor(2).to(torch.int64))   # (N,)
        occ_mov[occ_ms[:, 0], occ_ms[:, 1], occ_ms[:, 2]] = write_val
        occ_sfl = torch.zeros((3, 128, 128, 16), dtype=torch.float32)
        occ_sfl[:, occ_ms[:, 0], occ_ms[:, 1], occ_ms[:, 2]] = occ_sf.transpose(0, 1)
        occ_mask = torch.zeros((128, 128, 16), dtype=torch.bool, device=occ_sfl.device)
        occ_mask[occ_ms[:, 0], occ_ms[:, 1], occ_ms[:, 2]] = True
        # im pad
        im0 = F.pad(im0, pad=(0, 0, 0, 16), mode="constant", value=0)
        im1 = F.pad(im1, pad=(0, 0, 0, 16), mode="constant", value=0)

        return im0, im1, rasf0, rasc0, rapf0, rasf1, rasc1, rapf1, occ_mov, occ_sfl, occ_mask, self.files[idx]
    
def radar_sparse_tensor(rasf, rasc, params, device):
    # init
    bsr, N, cr = rasf.shape
    # idx in ran azi ele
    polar_idx = rasc.reshape(-1, 3) # [bsr*N,3]
    # batch
    batch_idx = torch.arange(bsr, device=device).view(-1,1).expand(-1,N)  # [B,N]
    batch_idx = batch_idx.reshape(-1,1)  # [B*N,1]
    # concat batch index -> N x 4
    coords = torch.cat([batch_idx, polar_idx], dim=1).to(torch.int32)  # [N,4]
    features = rasf.reshape(-1, cr)  # [N, cr]
    spatial_shape = [len(params['arrR']), len(params['arrA']), len(params['arrE'])]
    batch_size = bsr
    coords = coords.to(device)
    features = features.to(device)
    # SparseConvTensor
    sparse_tensor = spconv.SparseConvTensor(features, coords, spatial_shape, batch_size)

    return sparse_tensor
        
    