import datetime
import os
from utils import create_output_dir, create_dataset, load_config, joint2traj, recover_traj, loc2traj, batch_process_coords
from models import create_loc_model, create_traj_model
import sys
sys.path.append('..')
from train.losses import compute_ADE_FDE
from data import KeypointsDataset
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from itertools import chain
import time
import copy
from collections import defaultdict


class Evaluator:
    def __init__(self, args):

        self.eval_mode = args.eval_mode
        self.r_seed = args.r_seed
        self.bs = args.bs
        self.obs = args.obs
        self.pred = args.pred
        self.loc_cfg_path = args.loc_cfg
        self.traj_cfg_path = args.traj_cfg
        self.joints_folder = args.joints_folder
        self.load_loc = args.load_loc
        self.load_traj = args.load_traj

        # select device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Device: ', self.device)
        
        # random seed
        random.seed(self.r_seed)
        torch.manual_seed(self.r_seed)
        np.random.seed(self.r_seed)
        if use_cuda:
            torch.cuda.manual_seed(self.r_seed)

        # load config
        if self.loc_cfg_path:
            self.loc_config = load_config(self.loc_cfg_path)
            if torch.cuda.is_available():
                self.loc_config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
            else:
                self.loc_config["DEVICE"] = "cpu"
        if self.traj_cfg_path:
            self.traj_config = load_config(self.traj_cfg_path)
            if torch.cuda.is_available():
                self.traj_config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
            else:
                self.traj_config["DEVICE"] = "cpu"
 


        # Dataloader
        print(">>> creating dataloaders")
        self.dic_jo = create_dataset(self.joints_folder)
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.dic_jo, phase=phase),
                                            batch_size=self.bs, shuffle=False) for phase in ['test']} #dict to store dataloaders
        self.dataset_sizes = {phase: len(KeypointsDataset(self.dic_jo, phase=phase))
                            for phase in ['test']}

        print('Sizes of the dataset: {}'.format(self.dataset_sizes))


        # create models
        if self.eval_mode in ["loc", "traj_pred"]:
            self.loc_model = create_loc_model(self.loc_config)
            if self.load_loc != "":
                self.loc_model.load_state_dict(torch.load(self.load_loc))
            print(">>> Localization model params: {:.3f}M".format(sum(p.numel() for p in self.loc_model.parameters()) / 1000000.0))

        if self.eval_mode in ["traj_pred"]:
            self.traj_model = create_traj_model(self.traj_config)
            if self.load_traj != "":
                self.traj_model.load_state_dict(torch.load(self.load_traj))
            print(">>> Trajectory prediction model params: {:.3f}M".format(sum(p.numel() for p in self.traj_model.parameters()) / 1000000.0))
            self.total_parameters = sum(p.numel() for p in self.loc_model.parameters()) + sum(p.numel() for p in self.traj_model.parameters())
            print(">>> Total params: {:.3f}M".format(self.total_parameters / 1000000.0))


    def evaluate_loc(self):

        # Average distance on training and test set after unnormalizing
        self.loc_model.eval()
        dataset = KeypointsDataset(self.dic_jo, phase='test')         

        size_eval = len(dataset) 
        size_eval_seq = size_eval * (self.obs + self.pred) 
        start = 0
        epoch_ade = 0
        epoch_fde = 0
        epoch_traj_count = 0


        with torch.no_grad():
            for end in range(self.bs, size_eval + self.bs, self.bs):
                end = end if end < size_eval else size_eval
                inputs, labels, _, _, ego_pose, camera_pose, traj_3d_ego, _, _ = dataset[start:end]
                
                labels = labels.to(self.device)
                batch_size, seq_length, _ = inputs.size()
                labels = labels.view(batch_size * seq_length, -1) # 3d localization (10)

                
                scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)

                traj_3d_ego = traj_3d_ego.to(self.device)
                scene_train_real_ped = scene_train_real_ped.to(self.loc_config["DEVICE"])
                scene_train_mask = scene_train_mask.to(self.loc_config["DEVICE"])
                padding_mask = padding_mask.to(self.loc_config["DEVICE"])

                scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                scene_train_mask = scene_train_mask[:,0,:,:]

                start = end

                # Forward pass
                outputs = self.loc_model(scene_train_real_ped, padding_mask)
                ade, fde, traj_count = compute_ADE_FDE(outputs, ego_pose, camera_pose, traj_3d_ego, batch_size)
                epoch_ade += ade
                epoch_fde += fde
                epoch_traj_count += traj_count
        

        epoch_ade = epoch_ade / epoch_traj_count
        epoch_fde = epoch_fde / epoch_traj_count
        print(f"ADE: {epoch_ade}, FDE: {epoch_fde}")


    def evaluate_traj_pred(self):

        self.loc_model.eval()
        self.traj_model.eval()

        # output lists
        batch_id = 0
        ade_batch = 0 
        fde_batch = 0
        obs_pred_ls = []
        gt_traj_ls = []
        total_samples = 0

        with torch.no_grad():

            for inputs, labels, _, _, ego_pose, camera_pose, traj_3d_ego, _, _ in self.dataloaders['test']:
                
                labels = labels.to(self.device)
                batch_size, seq_length, _ = inputs.size()
                labels = labels.view(batch_size * seq_length, -1) 

                scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)

                scene_train_real_ped = scene_train_real_ped.to(self.traj_config["DEVICE"])
                scene_train_mask = scene_train_mask.to(self.traj_config["DEVICE"])
                padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                scene_train_mask = scene_train_mask[:,0,:,:]

                # Testing: only input observation
                scene_train_real_ped_obs = scene_train_real_ped[:,:self.obs,:,:]
                outputs = self.loc_model(scene_train_real_ped, padding_mask) 

                scene_train_real_ped_obs = scene_train_real_ped[:,:self.obs,:,:]
                padding_mask[:,self.obs:] = True
                outputs = self.loc_model(scene_train_real_ped_obs, padding_mask)
                
                # traj
                traj_estimated_ls = recover_traj(outputs, ego_pose, camera_pose)
                
                scene_train_real_ped, scene_train_mask, padding_mask = loc2traj(traj_estimated_ls)
                scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt = loc2traj(traj_3d_ego)

                in_joints, in_masks, out_joints, out_masks, padding_mask, _ = batch_process_coords(scene_train_real_ped, scene_train_mask, padding_mask, self.traj_config, training=False)
                in_joints_gt, in_masks_gt, out_joints_gt, out_masks_gt, padding_mask_gt, _ = batch_process_coords(scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt, self.traj_config, training=False)
                
                padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                pred_joints = self.traj_model(in_joints, padding_mask)

                pred_joints = pred_joints[:,-self.pred:]
                pred_joints = pred_joints.cpu()
                pred_joints = pred_joints + scene_train_real_ped[:,0:1,(self.obs-1):self.obs, 0, 0:2]
                out_joints = out_joints_gt.cpu() 

                out_joints = out_joints + scene_train_real_ped_gt[:,0:1,(self.obs-1):self.obs, 0, :]
                pred_joints = pred_joints.reshape(out_joints.size(0), self.pred, 1, 2)  

                # obs + pred
                # concatenate the observed and predicted trajectories
                pred_outputs = pred_joints.clone()
                # to numpy
                pred_outputs = pred_outputs.cpu().numpy()
                pred_outputs = pred_outputs[0,:,0,:]
                obs_pred = np.concatenate((traj_estimated_ls[0,:self.obs,:2], pred_outputs), axis=0)

                obs_pred_ls.append(obs_pred)
                gt_traj_ls.append(traj_3d_ego.cpu().numpy()[0,:,:2])


                for k in range(len(out_joints)):

                    person_out_joints = out_joints[k,:,0:1]
                    person_pred_joints = pred_joints[k,:,0:1]

                    gt_xy = person_out_joints[:,0,:2]
                    pred_xy = person_pred_joints[:,0,:2]
                    sum_ade = 0

                    for t in range(self.pred):
                        d1 = (gt_xy[t,0].detach().cpu().numpy() - pred_xy[t,0].detach().cpu().numpy())
                        d2 = (gt_xy[t,1].detach().cpu().numpy() - pred_xy[t,1].detach().cpu().numpy())
                    
                        dist_ade = [d1,d2]
                        sum_ade += np.linalg.norm(dist_ade)
                    
                    sum_ade /= self.pred
                    ade_batch += sum_ade
                    d3 = (gt_xy[-1,0].detach().cpu().numpy() - pred_xy[-1,0].detach().cpu().numpy())
                    d4 = (gt_xy[-1,1].detach().cpu().numpy() - pred_xy[-1,1].detach().cpu().numpy())
                    dist_fde = [d3,d4]
                    scene_fde = np.linalg.norm(dist_fde)

                    fde_batch += scene_fde
                    total_samples += 1
                
                batch_id+=1

            ade_avg = ade_batch/total_samples
            fde_avg = fde_batch/total_samples

            print(f"ADE: {ade_avg}, FDE: {fde_avg}")
          
            return ade_avg, fde_avg