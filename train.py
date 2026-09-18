from ruamel.yaml import YAML
params_path = 'Config/params.yaml'
# load Yaml
yaml_obj = YAML()
yaml_obj.preserve_quotes = True
with open(params_path, 'r') as file:
    params = yaml_obj.load(file)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = params['exp']['cuda_device']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ["MAX_JOBS"] = "1"
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

    def forward(self, im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf):
        return self.net(im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf)

    # train
    def training_step(self, batch, batch_idx):
        # data
        im0, im1, ra0_sp, ra1_sp, ra0_pf, ra1_pf, occ_mov, occ_sfl, occ_mask = self._extract_data(batch)
        # forward
        occ_ms, occ_sf, intern = self.forward(im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf)
        loss, loss_stem = self.loss_obj(occ_ms, occ_sf, intern, occ_mov, occ_sfl, occ_mask)
        # log per batch
        self.log_dict({f"train/{k}": v.item() for k, v in loss_stem.items()}, on_step=True, on_epoch=True)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # val
    def validation_step(self, batch, batch_idx):
        # data
        im0, im1, ra0_sp, ra1_sp, ra0_pf, ra1_pf, occ_mov, occ_sfl, occ_mask = self._extract_data(batch)
        # forward
        occ_ms, occ_sf, _ = self.forward(im0, im1, ra0_sp, ra0_pf, ra1_sp, ra1_pf)
        loss, loss_stem = self.val_loss_obj(occ_ms, occ_sf, occ_mov, occ_sfl, occ_mask)
        # log per batch
        self.log_dict({f"val/{k}": v.item() for k, v in loss_stem.items()}, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    # check
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"[Unused param] {name}")

    # opt + scheduler
    def configure_optimizers(self):
        opt = optim.Adam(self.net.parameters(), lr=self.params['train']['lr'])
        # Warmup + Cosine Scheduler
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.4, end_factor=1.0,
                                                   total_iters=self.params['train']['cosine'])
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.params['train']['epochs']-self.params['train']['cosine'], eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[self.params['train']['cosine']])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1,}}

    def _extract_data(self, data):
        # extract data
        im0, im1, rasf0, rasc0, rapf0, rasf1, rasc1, rapf1, occ_mov, occ_sfl, occ_mask = data
        # to cuda
        device = self.device
        im0, im1, ra0_pf, ra1_pf = im0.to(device), im1.to(device), rapf0.to(device), rapf1.to(device)
        ra0_sp = DataRead.radar_sparse_tensor(rasf0, rasc0, params, device)
        ra1_sp = DataRead.radar_sparse_tensor(rasf1, rasc1, params, device)
        occ_mov, occ_sfl, occ_mask = occ_mov.to(device), occ_sfl.to(device), occ_mask.to(device)
        return im0, im1, ra0_sp, ra1_sp, ra0_pf, ra1_pf, occ_mov, occ_sfl, occ_mask


# data
class RCFlowDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def setup(self, stage=None):
        self.train_dataset = DataRead.RCFlowDataset(self.params['path']['data_path'],
                                                    self.params['exp']['train_seq'],
                                                    self.params, mode='train')
        self.val_dataset = DataRead.RCFlowDataset(self.params['path']['data_path'],
                                                  self.params['exp']['val_seq'],
                                                  self.params, mode='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['train']['batch_size'],
                          shuffle=True, num_workers=self.params['exp']['num_workers'],
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params['train']['batch_size'],
                          shuffle=False, num_workers=self.params['exp']['num_workers'],
                          pin_memory=True, drop_last=True)

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

    # Lightning logger & checkpoint
    log_dir = os.path.join("Checkpoints", params['exp']['exp_name'])
    if os.path.exists(log_dir):
        print(f"[INFO] Log folder '{log_dir}' exists, removing it...")
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Created new log folder '{log_dir}'")
    logger = TensorBoardLogger(save_dir="Checkpoints", name=params['exp']['exp_name'], version=0)
    checkpoint_cb = ModelCheckpoint(dirpath=os.path.join("Checkpoints", params['exp']['exp_name'], "models"),
                                    save_top_k=3, monitor="val/loss", mode="min", filename="model-{epoch:02d}-{val/loss:.4f}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        max_epochs=params['train']['epochs'],
        strategy='ddp',
        precision=32,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    # data and model
    data_module = RCFlowDataModule(params)
    model = RCFlowModule(params)

    trainer.fit(model, datamodule=data_module)
