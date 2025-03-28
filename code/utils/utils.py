import os
import json
import yaml
import numpy as np
import torch

def load_config(path):
    """
    Load the config file and make any dynamic edits.
    """
    with open(path, "rt") as reader:
        config = yaml.load(reader, Loader=yaml.Loader)


    # # chceck path to the config file
    # if not os.path.exists(config["OUTPUT"]["ckpt_dir"]):
    #     os.makedirs(config["OUTPUT"]["ckpt_dir"], exist_ok=True)
    # with open(os.path.join(config["OUTPUT"]["ckpt_dir"], "config.yaml"), 'w') as f:
    #     yaml.dump(config, f)

    return config


def create_output_dir(output_dir, train_mode, exp_name, now_time):
    output_dir = os.path.join(output_dir, train_mode, exp_name + '_' + now_time)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def create_dataset_filter(joints_filter, joints_folder, obs):

    dic_jo = {'train':{'X':[], 'Y':[], 'names':[], 'kps':[], 'boxes_3d':[], 'boxes_2d':[], 'K':[], 'ego_pose':[], 'camera_pose':[], 'traj_3d_ego':[], 'image_path':[], 'traj_3d_fcos3d':[]}, \
              'test':{'X':[], 'Y':[], 'names':[], 'kps':[], 'boxes_3d':[], 'boxes_2d':[], 'K':[], 'ego_pose':[], 'camera_pose':[], 'traj_3d_ego':[],  'image_path':[], 'traj_3d_fcos3d':[]}}
    
    for split, filter_dis in zip(['train', 'test'], joints_filter):
        path = os.path.join(joints_folder, split)
        list_files = os.listdir(path)
        for file in list_files:
            with open(os.path.join(path, file), 'r') as f:
                dic = json.load(f)
                # check filter
                clst_ls_obs = dic['clst_ls'][:obs]
                if filter_dis == '30+':
                    # contain at least one 30+
                    if '30+' in clst_ls_obs:
                        pass
                    else:
                        continue
                elif filter_dis == '30':
                    # contain at least one 30 but no 30+
                    if '30' in clst_ls_obs and '30+' not in clst_ls_obs:
                        pass
                    else:
                        continue
                elif filter_dis == '20':
                    # contain at least one 20 but no 30+ and 30
                    if '20' in clst_ls_obs and '30+' not in clst_ls_obs and '30' not in clst_ls_obs:
                        pass
                    else:
                        continue
                elif filter_dis == '10':
                    # contain all 10
                    if clst_ls_obs == ['10', '10', '10', '10']:
                        pass
                    else:
                        continue
                elif filter_dis == 'all':
                    pass

                # append to the dictionary
                dic_jo[split]['X'].append(dic['X'])
                dic_jo[split]['Y'].append(dic['Y'])
                dic_jo[split]['names'].append(dic['names'])
                dic_jo[split]['kps'].append(dic['kps'])
                dic_jo[split]['boxes_3d'].append(dic['boxes_3d'])
                dic_jo[split]['boxes_2d'].append(dic['boxes_2d'])
                dic_jo[split]['K'].append(dic['K'])
                dic_jo[split]['ego_pose'].append(dic['ego_pose'])
                dic_jo[split]['camera_pose'].append(dic['camera_pose'])
                dic_jo[split]['traj_3d_ego'].append(dic['traj_3d_ego'])
                dic_jo[split]['image_path'].append(dic['image_path'])


    return dic_jo


def create_dataset(joints_folder):
    
    dic_jo = {'train':{'X':[], 'Y':[], 'names':[], 'kps':[], 'boxes_3d':[], 'boxes_2d':[], 'K':[], 'ego_pose':[], 'camera_pose':[], 'traj_3d_ego':[], 'image_path':[], 'traj_3d_fcos3d':[]}, \
              'test':{'X':[], 'Y':[], 'names':[], 'kps':[], 'boxes_3d':[], 'boxes_2d':[], 'K':[], 'ego_pose':[], 'camera_pose':[], 'traj_3d_ego':[],  'image_path':[], 'traj_3d_fcos3d':[]}}
    
    for split in ['train', 'test']:
        path = os.path.join(joints_folder, split)
        list_files = os.listdir(path)
        for file in list_files:
            with open(os.path.join(path, file), 'r') as f:
                dic = json.load(f)

                # append to the dictionary
                dic_jo[split]['X'].append(dic['X'])
                dic_jo[split]['Y'].append(dic['Y'])
                dic_jo[split]['names'].append(dic['names'])
                dic_jo[split]['kps'].append(dic['kps'])
                dic_jo[split]['boxes_3d'].append(dic['boxes_3d'])
                dic_jo[split]['boxes_2d'].append(dic['boxes_2d'])
                dic_jo[split]['K'].append(dic['K'])
                dic_jo[split]['ego_pose'].append(dic['ego_pose'])
                dic_jo[split]['camera_pose'].append(dic['camera_pose'])
                dic_jo[split]['traj_3d_ego'].append(dic['traj_3d_ego'])
                dic_jo[split]['image_path'].append(dic['image_path'])


    return dic_jo


def joint2traj(joint_ls):
    # input: (batch, 10, 34) => (batch, 10, 17, 2)
    joint_ls = joint_ls.reshape(joint_ls.shape[0], joint_ls.shape[1], -1, 2)
    # padding zeros: (batch, 10, 17, 2) => (batch, 10, 17, 4)
    joint_ls = np.concatenate((joint_ls, np.zeros((joint_ls.shape[0], joint_ls.shape[1], joint_ls.shape[2], 2))), axis=3)
    joint_ls = np.expand_dims(joint_ls, axis=1) 
    scene_train_mask = np.ones(joint_ls.shape[:-1])

    num_people_list = []
    for i in range(joint_ls.shape[0]):
        num_people_list.append(torch.zeros(joint_ls[i].shape[1]))
    padding_mask = torch.nn.utils.rnn.pad_sequence(num_people_list, batch_first=True, padding_value=1).bool()
    # to tensor
    # forward
    joint_ls = torch.from_numpy(joint_ls).float()
    scene_train_mask = torch.from_numpy(scene_train_mask).float()

    return joint_ls, scene_train_mask, padding_mask


def save_checkpoint(model, optimizer, epoch, config, filename, path=None):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    torch.save(ckpt, os.path.join(path, filename))

