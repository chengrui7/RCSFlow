from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
params_path = 'Config/testparams.yaml'
# load Yaml
yaml_obj = YAML()
yaml_obj.preserve_quotes = True
with open(params_path, 'r') as file:
    params = yaml_obj.load(file)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = params['exp']['cuda_device']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from Models import RCOF
from Losses import RCOFLoss
from Utils import ParamLoader, DataRead

class RCFlowModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters() 
        self.params = params
        # model
        self.net = RCOF.RCOF(params)
        self.loss_obj = RCOFLoss.RCOFLoss(params)
        self.val_loss_obj = RCOFLoss.RCOFValLoss(params)
        self.test_loss_obj = RCOFLoss.RCOFTest(params)

    def forward(self, im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf):
        return self.net(im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf, eval_mode=True)

    def on_test_epoch_start(self):
        self.test_loss_obj.reset()
    
    # train
    def test_step(self, batch, batch_idx):
        # data
        im0, im1, ra0_sp, ra1_sp, ra0_pf, ra1_pf, occ_mov, occ_sfl, occ_mask, file_name = self._extract_data(batch)
        # forward
        occ_ms, occ_sf = self.forward(im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf)
        metric = self.test_loss_obj.cal_batch(occ_ms, occ_sf, occ_mov, occ_sfl, occ_mask)
        ParamLoader.save_vis(im0, im1, occ_mov, occ_sfl, occ_ms, occ_sf, metric, file_name[0])
        # print(metric)
    
    def on_test_epoch_end(self):
        metric = self.test_loss_obj.cal_end()
        print(metric)

    def _extract_data(self, data):
        # extract data
        im0, im1, rasf0, rasc0, rapf0, rasf1, rasc1, rapf1, occ_mov, occ_sfl, occ_mask, file_name = data
        # to cuda
        device = self.device
        im0, im1, ra0_pf, ra1_pf = im0.to(device), im1.to(device), rapf0.to(device), rapf1.to(device)
        ra0_sp = DataRead.radar_sparse_tensor(rasf0, rasc0, params, device)
        ra1_sp = DataRead.radar_sparse_tensor(rasf1, rasc1, params, device)
        occ_mov, occ_sfl, occ_mask = occ_mov.to(device), occ_sfl.to(device), occ_mask.to(device)
        return im0, im1, ra0_sp, ra1_sp, ra0_pf, ra1_pf, occ_mov, occ_sfl, occ_mask, file_name
    
# data
class RCFlowDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def setup(self, stage=None):
        self.test_dataset = DataRead.RCFlowDataset(self.params['path']['data_path'],
                                                   self.params['exp']['test_seq'],
                                                   self.params, mode='test')
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.params['test']['batch_size'],
                          shuffle=False, num_workers=self.params['exp']['num_workers'], 
                          pin_memory=True, drop_last=False)



if __name__ == "__main__":

    # other params
    arrR, arrA, arrE, arrD = ParamLoader.load_axis(params['path']['polararr_path'], params['path']['dopplerarr_path'])
    params['arrR'] = torch.tensor(arrR.copy(), dtype=torch.float)
    params['arrA'] = torch.tensor(arrA.copy(), dtype=torch.float)
    params['arrE'] = torch.tensor(arrE.copy(), dtype=torch.float)
    params['arrD'] = torch.tensor(arrD.copy(), dtype=torch.float)

    # cam calib
    camyaml_obj = YAML(); camyaml_obj.preserve_quotes = True
    with open(params['path']['calib_path'], 'r') as file:
        camparams = camyaml_obj.load(file)
    intrinsics, distortion, r_cam, tr_lid_cam, img_size = ParamLoader.get_matrices_from_dict_lc_calib(camparams)
    params['intrinsics'] = torch.tensor(intrinsics.copy(), dtype=torch.float)
    params['imgsize'] = torch.tensor(img_size, dtype=torch.long)

    # extrinscis
    TransRadar2Lidar = torch.eye(4)
    TransRadar2Lidar[:3, 3] = torch.tensor([-2.54,0.0,0.7])
    extrinsicsRadar2Cam, extrinsicsCam2Radar = ParamLoader.get_extrinsics(tr_lid_cam, TransRadar2Lidar.numpy())
    params['extrinsicsRadar2Cam'] = torch.tensor(extrinsicsRadar2Cam.copy(), dtype=torch.float)
    params['extrinsicsCam2Radar'] = torch.tensor(extrinsicsCam2Radar.copy(), dtype=torch.float)

    # Lightning logger 
    log_dir = os.path.join("Checkpoints", params['exp']['exp_name'] + '_test')
    if os.path.exists(log_dir):
        print(f"[INFO] Log folder '{log_dir}' exists, removing it...")
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Created new log folder '{log_dir}'")
    logger = TensorBoardLogger(save_dir="Checkpoints", name=params['exp']['exp_name'] + '_test', version=0)

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=32,
        enable_progress_bar=True,
        logger = logger
    )

    trainer.test(
        model=RCFlowModule(params),
        dataloaders=RCFlowDataModule(params),
        ckpt_path=params['path']['model_path'],
        weights_only=False
    )