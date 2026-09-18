import os
import scipy.io
import numpy as np
import yaml
import torch
import random
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

# Load Doppler, Range, Azimuth and Elevation Axis
def load_axis(polar_path, doppler_path, is_in_rad=True, is_with_doppler=True, is_reverse_ae = True):
    temp_values = scipy.io.loadmat(polar_path)
    arr_range = temp_values['arrRange']
    if is_in_rad:
        deg2rad = np.pi/180.
        arr_azimuth = temp_values['arrAzimuth']*deg2rad
        arr_elevation = temp_values['arrElevation']*deg2rad
    else:
        arr_azimuth = temp_values['arrAzimuth']
        arr_elevation = temp_values['arrElevation']
    _, num_0 = arr_range.shape
    _, num_1 = arr_azimuth.shape
    _, num_2 = arr_elevation.shape
    arr_range = arr_range.reshape((num_0,))
    arr_azimuth = arr_azimuth.reshape((num_1,))
    arr_elevation = arr_elevation.reshape((num_2,))
    if is_reverse_ae:
        arr_azimuth = np.flip(-arr_azimuth)
        arr_elevation = np.flip(-arr_elevation)
    a_start = (num_1 - 96) // 2
    e_start = (num_2 - 32) // 2
    arr_azimuth = arr_azimuth[a_start:a_start+96]
    arr_elevation = arr_elevation[e_start:e_start+32]
    arr_range = arr_range[2:130]
    if is_with_doppler:
        arr_doppler = scipy.io.loadmat(doppler_path)['arr_doppler']
        _, num_3 = arr_doppler.shape
        arr_doppler = arr_doppler.reshape((num_3,))
        return arr_range, arr_azimuth, arr_elevation, arr_doppler
    else:
        return arr_range, arr_azimuth, arr_elevation

def split_dataset(data_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
        data_dict (dict): The input dictionary with frame numbers as keys.
        train_ratio (float): Proportion of the dataset for training (default is 0.7).
        val_ratio (float): Proportion of the dataset for validation (default is 0.15).
        test_ratio (float): Proportion of the dataset for testing (default is 0.15).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Three dictionaries (train_set, val_set, test_set) containing the split data.
    """
    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    train_set, val_set, test_set = {}, {}, {}

    # Extract all keys (frame numbers) and shuffle them
    for seq, frame_dict in data_dict.items():

        keys = list(frame_dict.keys())
        random.seed(seed)
        random.shuffle(keys)

        # Calculate split indices
        total_seq_frames = len(keys)
        train_end = int(total_seq_frames * train_ratio)
        val_end = train_end + int(total_seq_frames * val_ratio)

        # Split keys
        train_seq_keys = keys[:train_end]
        val_seq_keys = keys[train_end:val_end]
        test_seq_keys = keys[val_end:]

        # Create sub-dictionaries for each split
        train_set[seq] = {key: frame_dict[key] for key in train_seq_keys}
        val_set[seq] = {key: frame_dict[key] for key in val_seq_keys}
        test_set[seq] = {key: frame_dict[key] for key in test_seq_keys}

    split_set = {
                    'train_set': train_set,
                    'val_set': val_set,
                    'test_set': test_set
                }

    return split_set

# Load Radar Cube Mat
def load_power(cube_path1, cube_path2, is_reverse_ae = True):

    power_path = cube_path1
    power0 = scipy.io.loadmat(power_path)
    power0 = power0["arrDREA"][:,:,:,:]/(1e+13)
    power0 = np.transpose(power0, (0, 1, 3, 2)) #DRAE
    if is_reverse_ae:
        power0 = np.flip(np.flip(power0, axis=2), axis=3)
    power0 = (power0 - np.min(power0)) / (np.max(power0) - np.min(power0))

    next_power_path = cube_path2
    power1 = scipy.io.loadmat(next_power_path)
    power1 = power1["arrDREA"][:,:,:,:]/(1e+13)
    power1 = np.transpose(power1, (0, 1, 3, 2)) #DRAE
    if is_reverse_ae:
        power1 = np.flip(np.flip(power1, axis=2), axis=3)
    power1 = (power1 - np.min(power1)) / (np.max(power1) - np.min(power1))

    cube_tensor0 = torch.tensor(power0).to(dtype=torch.float)
    cube_tensor1 = torch.tensor(power1).to(dtype=torch.float)

    return cube_tensor0, cube_tensor1

def getcoords(params):
    arrR, arrA, arrE = params['radar_FFT_arr']['arrRange'], \
                        params['radar_FFT_arr']['arrAzimuth'], params['radar_FFT_arr']['arrElevation']
    r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()

    # rae dimension
    r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
    r_coords = torch.linspace(r_min, r_max, r_dim).view(r_dim, 1, 1).expand(r_dim, a_dim, e_dim)
    a_coords = torch.linspace(a_min, a_max, a_dim).view(1, a_dim, 1).expand(r_dim, a_dim, e_dim)
    e_coords = torch.linspace(e_min, e_max, e_dim).view(1, 1, e_dim).expand(r_dim, a_dim, e_dim)
    coords = torch.stack([r_coords, a_coords, e_coords], dim=0)
    coords = coords.to(dtype=torch.float)
    coords = coords.to('cuda')
    return coords

def make_lr_lambda(warmup_epochs, decay_epochs, decay_rate):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            return decay_rate ** ((epoch - warmup_epochs) // decay_epochs)
    return lr_lambda


def get_matrices_from_dict_lc_calib(dict_values):
    # list_lc_calib_keys = [
    #     'fx', 'fy', 'px', 'py', \
    #     'k1', 'k2', 'k3', 'k4', 'k5', \
    #     'roll_c', 'pitch_c', 'yaw_c', \
    #     'roll_l', 'pitch_l', 'yaw_l', 'x_l', 'y_l', 'z_l'
    # ]
    intrinsics = np.array([
        [dict_values['fx'], 0.0, dict_values['px']],
        [0.0, dict_values['fy'], dict_values['py']],
        [0.0, 0.0, 1.0]
    ])
    distortion = np.array([
        dict_values['k1'], dict_values['k2'], dict_values['k3'], \
        dict_values['k4'], dict_values['k5']
    ]).reshape((-1,1))

    ### Processing rotation matrix via scipy ###
    try:
        yaw_c = dict_values['yaw_c']
        pitch_c = dict_values['pitch_c']
        roll_c = dict_values['roll_c']
        r_cam = (R.from_euler('zyx', [yaw_c, pitch_c, roll_c], degrees=True)).as_matrix()
    except:
        r_cam = (R.from_euler('zyx', [0.0, 0.0, 0.0], degrees=True)).as_matrix()

    img_h = dict_values['img_size_h']
    img_w = dict_values['img_size_w']
    img_size = (img_w, img_h)

    yaw_l = dict_values['yaw_ldr2cam']
    pitch_l = dict_values['pitch_ldr2cam']
    roll_l = dict_values['roll_ldr2cam']

    # print(yaw_l, pitch_l, roll_l)
    r_l = (R.from_euler('zyx', [yaw_l, pitch_l, roll_l], degrees=True)).as_matrix()
    ### Processing rotation matrix via scipy ###

    x_l = dict_values['x_ldr2cam']
    y_l = dict_values['y_ldr2cam']
    z_l = dict_values['z_ldr2cam']
    
    tr_lid_cam = np.concatenate([r_l, np.array([x_l,y_l,z_l]).reshape(-1,1)], axis=1)

    return intrinsics, distortion, r_cam, tr_lid_cam, img_size

def get_extrinsics(tr_lid_cam, TransRadar2Lidar):
    def inv_se3(T: np.ndarray) -> np.ndarray:
        assert T.shape == (4,4), "Input must be a 4x4 matrix"
        R = T[:3, :3]
        t = T[:3, 3]  # keep as column vector
        Tinv = np.eye(4)
        Tinv[:3, :3] = R.T
        Tinv[:3, 3] = -R.T @ t
        return Tinv
    LidarToCamera = np.insert(tr_lid_cam, 3, values=[0,0,0,1], axis=0)
    # trans from lidar to cam
    extrinsicsCam2Radar = np.matmul(inv_se3(TransRadar2Lidar), inv_se3(LidarToCamera))
    extrinsicsRadar2Cam = np.matmul(LidarToCamera, TransRadar2Lidar)
    return extrinsicsRadar2Cam, extrinsicsCam2Radar

def compute_flowcolor(flow, radius=100):
    def make_color_wheel():
        """Middlebury color wheel"""
        RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3), dtype=np.uint8)

        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
        col += RY
        # YG
        colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
        colorwheel[col:col+YG, 1] = 255
        col += YG
        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
        col += GC
        # CB
        colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0,CB)/CB)
        colorwheel[col:col+CB, 2] = 255
        col += CB
        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
        col += BM
        # MR
        colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0,MR)/MR)
        colorwheel[col:col+MR, 0] = 255
        return colorwheel

    """Middlebury"""
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]
    u = flow[:, 0]
    v = flow[:, 1]
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols-1)
    k0 = np.floor(fk).astype(int)
    k1 = (k0+1) % ncols
    f = fk - k0

    img = np.zeros((u.shape[0], 3), dtype=np.uint8)

    for i in range(3):  # R,G,B
        col0 = colorwheel[k0, i] / 255.0
        col1 = colorwheel[k1, i] / 255.0
        col = (1-f)*col0 + f*col1
 
        col = 1 - rad/(10+rad) * (1-col)
        img[:, i] = np.floor(255*col)
        # img[:, i] = col

    y, x = np.mgrid[-radius:radius, -radius:radius]
    rad = np.sqrt(x**2 + y**2)
    a = np.arctan2(-y, -x) / np.pi
    fk = (a+1) / 2 * (ncols-1)

    k0 = np.floor(fk).astype(int)
    k1 = (k0+1) % ncols
    f = fk - k0

    img2 = np.zeros((2*radius, 2*radius, 3), dtype=np.uint8)
    for i in range(3):
        col0 = colorwheel[k0, i] / 255.0
        col1 = colorwheel[k1, i] / 255.0
        col = (1-f)*col0 + f*col1
        col = 1 - np.clip(rad/radius,0,1) * (1-col)   # 半径归一化
        img2[..., i] = np.floor(255*col)

    img2[rad>radius] = 255  # 外圈设为白色

    return img/255.0, img2  # shape (N,3)

def compute_depthcolor(depth):
    # depth = 1/depth
    vmin, vmax = np.min(depth), np.max(depth)
    print(f'vmin:{vmin}, vmax:{vmax}')
    cmap = plt.get_cmap('rainbow_r')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(depth))
    bgr = (rgba[:, :3][:, ::-1] * 255).astype(np.int32)
    return bgr

def compute_preddepthcolor(depth, depth_label):
    # depth = 1/depth
    vmin, vmax = np.min(depth_label), np.max(depth_label)
    cmap = plt.get_cmap('rainbow_r')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(depth))
    bgr = (rgba[:, :3][:, ::-1] * 255).astype(np.int32)
    return bgr

def save_vis(im0, im1, occ_mov, occ_sfl, occ_ms, occ_sf, metric, file_name):
    # image
    im0 = im0.squeeze()
    im0 = im0.cpu()
    im0 = im0.byte().permute(1, 2, 0).numpy()
    im1 = im1.squeeze()
    im1 = im1.cpu()
    im1 = im1.byte().permute(1, 2, 0).numpy()
    # gt ms
    idx_static = (occ_mov.squeeze() == 1).nonzero(as_tuple=False)   # N, 3
    idx_moving = (occ_mov.squeeze() == 2).nonzero(as_tuple=False)   # N, 3
    idx_all = torch.cat([idx_static, idx_moving], dim=0)
    idx_static = idx_static.detach().cpu().numpy().copy()
    idx_moving = idx_moving.detach().cpu().numpy().copy()
    idx_all = idx_all.detach().cpu().numpy().copy()
    # gt sf
    occ_sfl = occ_sfl.squeeze(0).permute(1, 2, 3, 0)
    flow_gt = occ_sfl[idx_all[:, 0], idx_all[:, 1], idx_all[:, 2]]
    flow_gt = flow_gt.detach().cpu().numpy().copy()

    # pred ms
    occ_ms = occ_ms.argmax(dim=1)
    idx_staticpred = (occ_ms.squeeze() == 1).nonzero(as_tuple=False)   # N, 3
    idx_movingpred = (occ_ms.squeeze() == 2).nonzero(as_tuple=False)   # N, 3
    idx_allpred = torch.cat([idx_staticpred, idx_movingpred], dim=0)
    idx_staticpred = idx_staticpred.detach().cpu().numpy().copy()
    idx_movingpred = idx_movingpred.detach().cpu().numpy().copy()
    idx_allpred = idx_allpred.detach().cpu().numpy().copy()
    # pred sf
    occ_sf = occ_sf.squeeze(0).permute(1, 2, 3, 0)
    flow_pred = occ_sf[idx_allpred[:, 0], idx_allpred[:, 1], idx_allpred[:, 2]]
    flow_pred = flow_pred.detach().cpu().numpy().copy()

    save_file = dict(
        im0 = im0, im1 = im1,
        gt_idx = idx_all, gt_mov = idx_moving, gt_sta = idx_static,
        gt_sf = flow_gt,
        pred_idx = idx_allpred, pred_mov = idx_movingpred, pred_sta = idx_staticpred,
        pred_sf = flow_pred)
    metric = "_".join([f"{k}-{v:.4f}" for k, v in metric.items()])
    savepath_dir = os.path.splitext(os.path.basename(file_name))[0] + '_' + metric
    savepath_path = 'Vis/' + savepath_dir + '.npy'
    np.save(savepath_path, save_file)
    return None