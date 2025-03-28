import datetime
import os
from utils import create_output_dir, create_dataset, load_config, joint2traj, recover_traj, loc2traj, batch_process_coords, save_checkpoint
from models import create_loc_model, create_traj_model
from data import KeypointsDataset
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from itertools import chain
import time
from .losses import CompositeLoss, MultiTaskLoss, compute_dir_loss, compute_ADE_FDE, compute_traj_loss
import copy
from collections import defaultdict


class Trainer:

    def __init__(self, args):

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        
        self.tasks = ('d', 'x', 'y', 'h', 'w', 'l', 'ori')
        self.lambdas = (1, 1, 1, 1, 1, 1, 1)

        self.joints_folder = args.joints_folder
        self.train_mode = args.train_mode
        self.epochs = args.epochs
        self.bs = args.bs
        self.r_seed = args.r_seed
        self.no_save = args.no_save
        self.save_every = args.save_every
        self.eval_every = args.eval_every
        self.output_dir = args.output_dir
        self.exp_name = args.exp_name
        
        self.dir_loss = args.dir_loss
        self.dir_loss_w = args.dir_loss_w
        self.loc_cfg_path = args.loc_cfg
        self.seq_len = args.seq_len
        self.epochs_loc = args.epochs_loc

        self.obs = args.obs
        self.pred = args.pred
        self.traj_cfg_path = args.traj_cfg
        self.epochs_traj = args.epochs_traj
        self.gt = False
        self.val_gt = args.val_gt

        self.load_loc = args.load_loc
        self.load_traj = args.load_traj

        self.w_loc = args.w_loc
        self.w_traj = args.w_traj
        
        # path to save model and log file
        if not self.no_save:
            self.output_dir = create_output_dir(self.output_dir, self.train_mode, self.exp_name, now_time)
        
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

        # create dataset
        self.dic_jo = create_dataset(self.joints_folder)
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.dic_jo, phase=phase),
                                            batch_size=self.bs, shuffle=False) for phase in ['train', 'test']} 
    
        self.dataset_sizes = {phase: len(KeypointsDataset(self.dic_jo, phase=phase))
                            for phase in ['train', 'test']}
        
        print('Dataset: ', self.dataset_sizes)

        # create models
        if self.train_mode in ["joint", "freeze_loc", "loc"]:
            self.loc_model = create_loc_model(self.loc_config)
            if self.load_loc != "":
                self.loc_model.load_state_dict(torch.load(self.load_loc))
            print(">>> Localization model params: {:.3f}M".format(sum(p.numel() for p in self.loc_model.parameters()) / 1000000.0))

        if self.train_mode in ["joint", "freeze_loc"]:
            self.traj_model = create_traj_model(self.traj_config)
            if self.load_traj != "":
                self.traj_model.load_state_dict(torch.load(self.load_traj))
            print(">>> Trajectory prediction model params: {:.3f}M".format(sum(p.numel() for p in self.traj_model.parameters()) / 1000000.0))
            self.total_parameters = sum(p.numel() for p in self.loc_model.parameters()) + sum(p.numel() for p in self.traj_model.parameters())
            print(">>> Total params: {:.3f}M".format(self.total_parameters / 1000000.0))

        # create loss function for localization
        if self.train_mode in ["joint", "loc"]:
            losses_tr, losses_val = CompositeLoss(self.tasks)()
            self.mt_loss = MultiTaskLoss(losses_tr, losses_val, self.lambdas, self.tasks)
            self.mt_loss.to(self.device)
            print(">>> loss params: {}".format(sum(p.numel() for p in self.mt_loss.parameters())))

        # create optimizer
        if self.train_mode == 'joint':
            params_loc = self.loc_model.parameters()
            params_traj = self.traj_model.parameters()
            self.optimizer_loc = torch.optim.Adam(params=params_loc, lr=float(self.loc_config['TRAIN']['lr']))
            self.optimizer_traj = torch.optim.Adam(params=params_traj, lr=float(self.traj_config['TRAIN']['lr']))
        elif self.train_mode == 'freeze_loc':
            params_traj = self.traj_model.parameters()
            self.optimizer_traj = torch.optim.Adam(params=params_traj, lr=float(self.traj_config['TRAIN']['lr']))
        elif self.train_mode == 'loc':
            params_loc = chain(self.loc_model.parameters(), self.mt_loss.parameters())
            self.optimizer_loc = torch.optim.Adam(params=params_loc, lr=float(self.loc_config['TRAIN']['lr']))


    
    def train_joint(self):
        print("Joint training")
        since = time.time()
        best_loss_traj = 1e6
        best_epoch_traj = 0

        for epoch in range(self.epochs_traj):
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.loc_model.train()
                    self.traj_model.train()
                else:
                    total_traj_loss = 0
                    self.loc_model.eval()  
                    self.traj_model.eval()

                for inputs, labels, _, _, ego_pose, camera_pose, traj_3d_ego, _, _ in self.dataloaders[phase]:

                    labels = labels.to(self.device)
                    batch_size, seq_length, _ = inputs.size()
                    labels = labels.view(batch_size * seq_length, -1) 

                    scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)

                    scene_train_real_ped = scene_train_real_ped.to(self.traj_config["DEVICE"])
                    scene_train_mask = scene_train_mask.to(self.traj_config["DEVICE"])
                    padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                    scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                    scene_train_mask = scene_train_mask[:,0,:,:]

                    padding_mask[:, self.obs:] = True
                    

                    if phase == 'train':
                        with torch.set_grad_enabled(phase == 'train'):
                            
                            self.optimizer_loc.zero_grad()
                            self.optimizer_traj.zero_grad()

                            outputs = self.loc_model(scene_train_real_ped, padding_mask)
                            loss_monoloco, _ = self.mt_loss(outputs, labels, phase=phase)

                            if self.dir_loss: 
                                loss_dir, _ = compute_dir_loss(outputs, ego_pose, camera_pose, traj_3d_ego, self.obs, self.pred, self.device)
                                loss_dir = (loss_dir * self.dir_loss_w)
                                loss_monoloco += loss_dir


                            traj_estimated_ls = recover_traj(outputs, ego_pose, camera_pose)
                            scene_train_real_ped, scene_train_mask, padding_mask = loc2traj(traj_estimated_ls)
                            scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt = loc2traj(traj_3d_ego)
                            # estimate
                            in_joints, in_masks, out_joints, out_masks, padding_mask, _ = batch_process_coords(scene_train_real_ped, scene_train_mask, padding_mask, self.traj_config, training=False)
                            # GT
                            in_joints_gt, in_masks_gt, out_joints_gt, out_masks_gt, padding_mask_gt, _ = batch_process_coords(scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt, self.traj_config, training=False)
                            
                            padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                            diff = scene_train_real_ped_gt[:,0,self.obs-1:self.obs,:,:] - scene_train_real_ped[:,0,self.obs-1:self.obs,:,:]
  
                            if self.gt:
                                loss_traj, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints_gt, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None)
                            else:
                                loss_traj, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None)

                            loss_total = self.w_loc * loss_monoloco + self.w_traj * loss_traj

                            loss_total.backward()
                            torch.nn.utils.clip_grad_norm_(self.loc_model.parameters(), self.loc_config["TRAIN"]["max_grad_norm"])
                            torch.nn.utils.clip_grad_norm_(self.traj_model.parameters(), self.traj_config["TRAIN"]["max_grad_norm"])

                            self.optimizer_loc.step()
                            self.optimizer_traj.step()

                    else: # validation
                        with torch.no_grad():
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
                            diff = scene_train_real_ped_gt[:,0,self.obs-1:self.obs,:,:] - scene_train_real_ped[:,0,self.obs-1:self.obs,:,:]
                            out_joints_gt = diff.to(self.traj_config["DEVICE"]) + out_joints_gt 

                            if self.val_gt:
                                loss_traj_eval, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints_gt, in_masks, out_masks, padding_mask, epoch=epoch, mode='val', optimizer=None)
                            else:
                                loss_traj_eval, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=epoch, mode='val', optimizer=None)

                            total_traj_loss += (loss_traj_eval / 100)


            # val trajectory
            total_traj_loss = total_traj_loss / len(self.dataloaders['test'])
            if total_traj_loss < best_loss_traj:
                best_loss_traj = total_traj_loss
                best_epoch_traj = epoch
                print('------------------------------BEST MODEL UPDATED------------------------------')
                print('Best ADE: ', best_loss_traj, ' at epoch: ', best_epoch_traj)
                # save the model
                best_loc_model_wts = copy.deepcopy(self.loc_model.state_dict())
                best_traj_model_wts = copy.deepcopy(self.traj_model.state_dict())
                
                torch.save(self.loc_model.state_dict(), os.path.join(self.output_dir, 'best_loc_model.pth'))
                torch.save(self.traj_model.state_dict(), os.path.join(self.output_dir, 'best_traj_model.pth'))

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        print('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        print('Best validation ADE: {:.3f}'.format(best_loss_traj))
        print('Saved weights of the model at epoch: {}'.format(best_epoch_traj))
        # load best model weights
        self.loc_model.load_state_dict(best_loc_model_wts)
        self.traj_model.load_state_dict(best_traj_model_wts)

        return best_epoch_traj

    
    def train_freeze_loc(self):
        print("Freeze localization")
        since = time.time()
        best_loss_traj = 1e6
        best_epoch_traj = 0

        for epoch in range(self.epochs_traj):
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.loc_model.eval()  
                    self.traj_model.train()
                else:
                    total_traj_loss = 0
                    self.loc_model.eval()  
                    self.traj_model.eval()

                for inputs, labels, _, _, ego_pose, camera_pose, traj_3d_ego, _, _ in self.dataloaders[phase]:

                    labels = labels.to(self.device)
                    batch_size, seq_length, _ = inputs.size()
                    labels = labels.view(batch_size * seq_length, -1) 

                    scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)

                    scene_train_real_ped = scene_train_real_ped.to(self.traj_config["DEVICE"])
                    scene_train_mask = scene_train_mask.to(self.traj_config["DEVICE"])
                    padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                    scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                    scene_train_mask = scene_train_mask[:,0,:,:]

                    padding_mask[:, self.obs:] = True
                    outputs = self.loc_model(scene_train_real_ped, padding_mask)

                    if phase == 'train':
                        with torch.set_grad_enabled(phase == 'train'):
                            self.optimizer_traj.zero_grad()
                            traj_estimated_ls = recover_traj(outputs, ego_pose, camera_pose)
                            scene_train_real_ped, scene_train_mask, padding_mask = loc2traj(traj_estimated_ls)
                            scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt = loc2traj(traj_3d_ego)
                            # estimate
                            in_joints, in_masks, out_joints, out_masks, padding_mask, _ = batch_process_coords(scene_train_real_ped, scene_train_mask, padding_mask, self.traj_config, training=False)
                            # GT
                            in_joints_gt, in_masks_gt, out_joints_gt, out_masks_gt, padding_mask_gt, _ = batch_process_coords(scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt, self.traj_config, training=False)
                            
                            padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                            diff = scene_train_real_ped_gt[:,0,self.obs-1:self.obs,:,:] - scene_train_real_ped[:,0,self.obs-1:self.obs,:,:]
  
                            if self.gt:
                                loss_traj, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints_gt, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None)
                            else:
                                loss_traj, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None)

                            loss_traj.backward()
                            torch.nn.utils.clip_grad_norm_(self.traj_model.parameters(), self.traj_config["TRAIN"]["max_grad_norm"])
                            self.optimizer_traj.step()

                    else: # validation
                        with torch.no_grad():
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
                            diff = scene_train_real_ped_gt[:,0,self.obs-1:self.obs,:,:] - scene_train_real_ped[:,0,self.obs-1:self.obs,:,:]
                            out_joints_gt = diff.to(self.traj_config["DEVICE"]) + out_joints_gt 

                            if self.val_gt:
                                loss_traj_eval, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints_gt, in_masks, out_masks, padding_mask, epoch=epoch, mode='val', optimizer=None)
                            else:
                                loss_traj_eval, _ = compute_traj_loss(self.traj_model, self.traj_config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=epoch, mode='val', optimizer=None)

                            total_traj_loss += (loss_traj_eval / 100)

        

            
            # val trajectory
            total_traj_loss = total_traj_loss / len(self.dataloaders['test'])
            if total_traj_loss < best_loss_traj:
                best_loss_traj = total_traj_loss
                best_epoch_traj = epoch
                print('------------------------------BEST MODEL UPDATED------------------------------')
                print('Best ADE: ', best_loss_traj, ' at epoch: ', best_epoch_traj)
                # save the model
                best_model_wts = copy.deepcopy(self.traj_model.state_dict())
                torch.save(self.traj_model.state_dict(), os.path.join(self.output_dir, 'best_traj_model.pth'))

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        print('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        print('Best validation ADE: {:.3f}'.format(best_loss_traj))
        print('Saved weights of the model at epoch: {}'.format(best_epoch_traj))
        # load best model weights
        self.traj_model.load_state_dict(best_model_wts)

        return best_epoch_traj


    def train_loc(self):
        print("Localization")
        since = time.time()
        best_model_wts = copy.deepcopy(self.loc_model.state_dict())
        best_ade = 1e6
        epoch_ade = 0
        best_epoch = 0
        ade_count = 0
        epoch_losses = defaultdict(lambda: defaultdict(list))

        for epoch in range(self.epochs_loc):

            running_loss = defaultdict(lambda: defaultdict(int))

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.loc_model.train() 
                else:
                    self.loc_model.eval()  
                    epoch_ade = 0
                    ade_count = 0

                for inputs, labels, _, _, ego_pose, camera_pose, traj_3d_ego, _, _ in self.dataloaders[phase]:

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

                    if phase == 'train':
                        with torch.set_grad_enabled(phase == 'train'):
                            self.optimizer_loc.zero_grad()

                            outputs = self.loc_model(scene_train_real_ped, padding_mask)
                            loss_monoloco, _ = self.mt_loss(outputs, labels, phase=phase)

                            if self.dir_loss: 
                                loss_dir, _ = compute_dir_loss(outputs, ego_pose, camera_pose, traj_3d_ego, self.obs, self.pred, self.device)
                                loss_dir = (loss_dir * self.dir_loss_w)
                                loss_monoloco += loss_dir
                            

                            loss_monoloco.backward()
                            torch.nn.utils.clip_grad_norm_(self.loc_model.parameters(), self.loc_config["TRAIN"]["max_grad_norm"])
                            self.optimizer_loc.step()

                    # no grad
                    else:
                        with torch.no_grad():
                            outputs = self.loc_model(scene_train_real_ped, padding_mask)


                    with torch.no_grad():
                        loss_eval, loss_values_eval = self.mt_loss(outputs, labels, phase='val')
                        self.epoch_logs(phase, loss_eval, loss_values_eval, inputs, running_loss)
                        # validata ADE
                        if phase == 'test':
                            ade, fde, traj_count = compute_ADE_FDE(outputs, ego_pose, camera_pose, traj_3d_ego, batch_size)
                            epoch_ade += ade
                            ade_count += traj_count

           
            # print 
            self.cout_values(epoch, epoch_losses, running_loss)

            epoch_ade = epoch_ade / ade_count
            if epoch_ade < best_ade:
                best_ade = epoch_ade
                best_epoch = epoch
                best_model_wts = copy.deepcopy(self.loc_model.state_dict())
                print('------------------------------------------------BEST MODEL UPDATED------------------------------------------------')
                print('Best ADE: ', best_ade, ' at epoch: ', best_epoch)
                # save model
                torch.save(self.loc_model.state_dict(), os.path.join(self.output_dir, 'best_loc_model.pth'))
            
            epoch_ade = 0
            ade_count = 0
           

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        print('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        print('Best validation ADE: {:.3f}'.format(best_ade))
        print('Saved weights of the model at epoch: {}'.format(best_epoch))

        # load best model weights
        self.loc_model.load_state_dict(best_model_wts)

        return best_epoch
    

    def epoch_logs(self, phase, loss, loss_values, inputs, running_loss):

        running_loss[phase]['all'] += loss.item() * inputs.size(0)
        for i, task in enumerate(self.tasks):
            running_loss[phase][task] += loss_values[i].item() * inputs.size(0)


    def cout_values(self, epoch, epoch_losses, running_loss):

        string = '\r' + '{:.0f} '
        format_list = [epoch]
        for phase in running_loss:
            string = string + phase[0:1].upper() + ':'
            for el in running_loss['train']:
                loss = running_loss[phase][el] / self.dataset_sizes[phase]
                epoch_losses[phase][el].append(loss)
                if el == 'all':
                    string = string + ':{:.1f}  '
                    format_list.append(loss)
                elif el in ('ori', 'aux'):
                    string = string + el + ':{:.1f}  '
                    format_list.append(loss)
                else:
                    string = string + el + ':{:.0f}  '
                    format_list.append(loss * 100)

        if epoch % 10 == 0: # print every 10 epochs
            print(string.format(*format_list))
