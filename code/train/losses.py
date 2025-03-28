"""
Adapted from https://github.com/openpifpaf,
which is: 'Copyright 2019-2021 by Sven Kreiss and contributors. All rights reserved.'
and licensed under GNU AGPLv3
"""


import math
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from pytorch3d.transforms import quaternion_to_matrix
from utils import extract_labels, extract_labels_aux, extract_outputs, extract_outputs_no_detach



class MultiTaskLoss(torch.nn.Module):
    def __init__(self, losses_tr, losses_val, lambdas, tasks):
        super().__init__()

        self.losses = torch.nn.ModuleList(losses_tr)
        self.losses_val = losses_val
        self.lambdas = lambdas
        self.tasks = tasks
        if len(self.tasks) == 1 and self.tasks[0] == 'aux':
            self.flag_aux = True
        else:
            self.flag_aux = False

    def forward(self, outputs, labels, phase='train'):

        assert phase in ('train', 'val')
        out = extract_outputs(outputs, tasks=self.tasks)
        if self.flag_aux:
            gt_out = extract_labels_aux(labels, tasks=self.tasks)
        else:
            gt_out = extract_labels(labels, tasks=self.tasks)
        loss_values = [lam * l(o, g) for lam, l, o, g in zip(self.lambdas, self.losses, out, gt_out)]
        loss = sum(loss_values)

        if phase == 'val':
            loss_values_val = [l(o, g) for l, o, g in zip(self.losses_val, out, gt_out)]
            return loss, loss_values_val
        return loss, loss_values


class CompositeLoss(torch.nn.Module):

    def __init__(self, tasks):
        super().__init__()

        self.tasks = tasks
        self.multi_loss_tr = {task: (LaplacianLoss() if task == 'd'
                                     else (nn.BCEWithLogitsLoss() if task in ('aux', )
                                           else nn.L1Loss())) for task in tasks}
        self.multi_loss_val = {}

        for task in tasks:
            if task == 'd':
                loss = l1_loss_from_laplace
            elif task == 'ori':
                loss = angle_loss
            elif task in ('aux', ):
                loss = nn.BCEWithLogitsLoss()
            else:
                loss = nn.L1Loss()
            self.multi_loss_val[task] = loss

    def forward(self):
        losses_tr = [self.multi_loss_tr[l] for l in self.tasks]
        losses_val = [self.multi_loss_val[l] for l in self.tasks]
        return losses_tr, losses_val


class LaplacianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance"""
    def __init__(self, size_average=True, reduce=True, evaluate=False):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate

    def laplacian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py

        """
        eps = 0.01  # To avoid 0/0 when no uncertainty
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]
        norm = 1 - mu / xx  # Relative
        const = 2

        term_a = torch.abs(norm) * torch.exp(-si) + eps
        term_b = si
        norm_bi = (np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(torch.exp(si).cpu().detach().numpy()))

        if self.evaluate:
            return norm_bi
        return term_a + term_b + const

    def forward(self, outputs, targets):

        values = self.laplacian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


class GaussianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """
    def __init__(self, device, size_average=True, reduce=True, evaluate=False):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate
        self.device = device

    def gaussian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]

        min_si = torch.ones(si.size()).cuda(self.device) * 0.1
        si = torch.max(min_si, si)
        norm = xx - mu
        term_a = (norm / si)**2 / 2
        term_b = torch.log(si * math.sqrt(2 * math.pi))

        norm_si = (np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(si.cpu().detach().numpy()))

        if self.evaluate:
            return norm_si

        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.gaussian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


class CustomL1Loss(torch.nn.Module):
    """
    Experimental, not used.
    L1 loss with more weight to errors at a shorter distance
    It inherits from nn.module so it supports backward
    """

    def __init__(self, dic_norm, device, beta=1):
        super().__init__()

        self.dic_norm = dic_norm
        self.device = device
        self.beta = beta

    @staticmethod
    def compute_weights(xx, beta=1):
        """
        Return the appropriate weight depending on the distance and the hyperparameter chosen
        alpha = 1 refers to the curve of A Photogrammetric Approach for Real-time...
        It is made for unnormalized outputs (to be more understandable)
        From 70 meters on every value is weighted the same (0.1**beta)
        Alpha is optional value from Focal loss. Yet to be analyzed
        """
        # alpha = np.maximum(1, 10 ** (beta - 1))
        alpha = 1
        ww = np.maximum(0.1, 1 - xx / 78)**beta

        return alpha * ww

    def print_loss(self):
        xx = np.linspace(0, 80, 100)
        y1 = self.compute_weights(xx, beta=1)
        y2 = self.compute_weights(xx, beta=2)
        y3 = self.compute_weights(xx, beta=3)
        plt.plot(xx, y1)
        plt.plot(xx, y2)
        plt.plot(xx, y3)
        plt.xlabel("Distance [m]")
        plt.ylabel("Loss function Weight")
        plt.legend(("Beta = 1", "Beta = 2", "Beta = 3"))
        plt.show()

    def forward(self, output, target):

        unnormalized_output = output.cpu().detach().numpy() * self.dic_norm['std']['Y'] + self.dic_norm['mean']['Y']
        weights_np = self.compute_weights(unnormalized_output, self.beta)
        weights = torch.from_numpy(weights_np).float().to(self.device)  # To make weights in the same cuda device
        losses = torch.abs(output - target) * weights
        loss = losses.mean()  # Mean over the batch
        return loss


def angle_loss(orient, gt_orient):
    """Only for evaluation"""
    angles = torch.atan2(orient[:, 0], orient[:, 1])
    gt_angles = torch.atan2(gt_orient[:, 0], gt_orient[:, 1])
    # assert all(angles < math.pi) & all(angles > - math.pi)
    # assert all(gt_angles < math.pi) & all(gt_angles > - math.pi)
    loss = torch.mean(torch.abs(angles - gt_angles)) * 180 / 3.14
    return loss


def l1_loss_from_laplace(out, gt_out):
    """Only for evaluation"""
    loss = torch.mean(torch.abs(out[:, 0:1] - gt_out))
    return loss


def directional_loss_batch(traj_pred, traj_gt):
    # Ensure the input trajectories are torch tensors
    if not isinstance(traj_pred, torch.Tensor):
        traj_pred = torch.tensor(traj_pred, dtype=torch.float32)
    if not isinstance(traj_gt, torch.Tensor):
        traj_gt = torch.tensor(traj_gt, dtype=torch.float32)

    # Calculate vectors between consecutive points for predicted trajectory
    vectors_pred = traj_pred[:, 1:] - traj_pred[:, :-1]
    vectors_gt = traj_gt[:, 1:] - traj_gt[:, :-1]


    # Calculate norms and add a small epsilon for numerical stability
    epsilon = 1e-8
    norms_pred = torch.norm(vectors_pred, dim=2, keepdim=True) + epsilon
    norms_gt = torch.norm(vectors_gt, dim=2, keepdim=True) + epsilon

    # Normalize these vectors to get unit vectors
    unit_vectors_pred = vectors_pred / norms_pred
    unit_vectors_gt = vectors_gt / norms_gt
    # Compute cosine similarity between corresponding vectors
    cosine_similarity = torch.sum(unit_vectors_pred * unit_vectors_gt, dim=2)

    # Loss is the mean of (1 - cosine similarity) across all batches and sequences
    loss = torch.mean(1 - cosine_similarity)

    return loss


def compute_dir_loss(outputs, ego_pose, camera_pose, traj_3d_ego, obs, pred, device):

    def local2global( traj_estimated, ego_pose, camera_pose):
        # Camera translation
        camera_translation = camera_pose[:, :, 4:]
        # Camera rotation
        camera_quaternion = camera_pose[:, :, :4]
        # Ego translation
        ego_translation = ego_pose[:, :, 4:]
        # Ego rotation
        ego_quaternion = ego_pose[:, :, :4]
        
        # Quaternion to rotation matrix
        camera_rotation_matrix = quaternion_to_matrix(camera_quaternion)
        ego_rotation_matrix = quaternion_to_matrix(ego_quaternion)

        # Reshape traj_estimated to [batch, seq_len, 3, 1] for matrix multiplication
        traj_estimated = traj_estimated.unsqueeze(-1)
        # Local to camera
        traj_estimated = torch.matmul(camera_rotation_matrix, traj_estimated)
        traj_estimated = traj_estimated.squeeze(-1) + camera_translation  # Remove the extra dimension then add
        # Add another unsqueeze for camera to global transformation
        traj_estimated = traj_estimated.unsqueeze(-1)
        # Camera to global
        traj_estimated = torch.matmul(ego_rotation_matrix, traj_estimated)
        traj_estimated = traj_estimated.squeeze(-1) + ego_translation  # Remove the extra dimension then add

        return traj_estimated



    ego_pose = ego_pose.to(device, dtype=torch.float32)
    camera_pose = camera_pose.to(device, dtype=torch.float32)
    traj_3d_ego = traj_3d_ego.to(device, dtype=torch.float32)

    outputs_calib = extract_outputs_no_detach(outputs)
    xyzd = outputs_calib['xyzd']

    # batch* seq_len, 4 => batch, seq_len, 4
    xyzd_input = xyzd.view(-1, obs + pred, 4)
    # local to global

    xyzd_global = local2global(xyzd_input[:,:,:3], ego_pose, camera_pose)
    dir_loss = directional_loss_batch(xyzd_global[:,:,:2], traj_3d_ego[:,:,:2])

    return dir_loss, xyzd_global


def compute_ADE_FDE(outputs, ego_pose, camera_pose, traj_3d_ego, seq_total):

    def rotate(x, quaternion):
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        x_rotate = np.dot(quaternion.rotation_matrix, x)
        return x_rotate

    outputs_calib = extract_outputs(outputs)

    # get 3D estimated trajectory 
    xyzd = outputs_calib['xyzd']
    # calibrate the trajectory
    count = 0
    seq_len = xyzd.size(0) // seq_total
    ade_total = 0
    fde_total = 0
    ade_ls = []
    fde_ls = []

    for i in range(0, xyzd.size(0), seq_len):

        traj_3d_ego_i = traj_3d_ego[count].cpu().numpy()
        ego_pose_i = ego_pose[count].cpu().numpy()
        camera_pose_i = camera_pose[count].cpu().numpy()
        traj_3d_ego_i_estimated = []
        # calibrate the trajectory
        traj_estimated = xyzd[i:i+seq_len].cpu().numpy()
        for j in range(seq_len):
            ego_rotation = ego_pose_i[j, :4]
            ego_translation = ego_pose_i[j, 4:]
            camera_rotation = camera_pose_i[j, :4]
            camera_translation = camera_pose_i[j, 4:]

            xyz = traj_estimated[j, :3]
            # local to global
            # camera
            camera_q = Quaternion(camera_rotation)
            xyz = rotate(xyz, camera_q)
            xyz = xyz + camera_translation
            # ego
            ego_q = Quaternion(ego_rotation)
            xyz = rotate(xyz, ego_q)
            xyz = xyz + ego_translation
            traj_3d_ego_i_estimated.append(xyz)
            
        traj_3d_ego_i_estimated = np.array(traj_3d_ego_i_estimated)
        # calculate the distance

        distances = np.linalg.norm(traj_3d_ego_i[:,:2] - traj_3d_ego_i_estimated[:,:2], axis=1)
        ade =  np.mean(distances)
        fde = np.linalg.norm(traj_3d_ego_i[-1,:2] - traj_3d_ego_i_estimated[-1,:2])
        

        ade_ls.append(ade)
        fde_ls.append(fde)
        count += 1

        # self.est_data.append([traj_3d_ego_i_estimated[:,0], traj_3d_ego_i_estimated[:,1]])

        ade_total += ade
        fde_total += fde

    
    ade_avg = ade_total / count
    fde_avg = fde_total / count

    return ade_total, fde_total, count


def compute_traj_loss(model, config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=None, mode='val', loss_last=True, optimizer=None):
    
    _, in_F, _, _ = in_joints.shape

    metamask = (mode == 'train')
    pred_joints = model(in_joints, padding_mask, metamask=metamask)

    loss = MSE_LOSS(pred_joints[:,in_F:], out_joints, out_masks)

    return loss, pred_joints


def MSE_LOSS(output, target, mask=None, check=False):

    pred_xy = output[:,:,0,:2]
    gt_xy = target[:,:,0,:2]


    norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)
    mean_K = torch.mean(norm, dim=-1)
    mean_B = torch.mean(mean_K)
    if check:
        breakpoint()

    return mean_B*100