
import torch
from torch.utils.data import Dataset

class KeypointsDataset(Dataset):
    """
    Dataloader fro nuscenes or kitti datasets
    """

    def __init__(self, dic_jo, phase):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        self.inputs_all = torch.tensor(dic_jo[phase]['X'])
        self.outputs_all = torch.tensor(dic_jo[phase]['Y'])
        self.names_all = dic_jo[phase]['names']
        self.kps_all = torch.tensor(dic_jo[phase]['kps'])
        self.ego_pose_all = torch.tensor(dic_jo[phase]['ego_pose'])
        self.camera_pose_all = torch.tensor(dic_jo[phase]['camera_pose'])
        self.traj_3d_ego =  torch.tensor(dic_jo[phase]['traj_3d_ego'])
        self.boxes_2d = torch.tensor(dic_jo[phase]['boxes_2d'])
        self.kk = torch.tensor(dic_jo[phase]['K'])

        self.traj_3d_fcos3d = None
        # check empty list

        if 'traj_3d_fcos3d' in dic_jo[phase].keys() and dic_jo[phase]['traj_3d_fcos3d'] != []:
            self.traj_3d_fcos3d = torch.tensor(dic_jo[phase]['traj_3d_fcos3d'])

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        kps = self.kps_all[idx, :]
        ego_pose = self.ego_pose_all[idx, :]
        camera_pose = self.camera_pose_all[idx, :]
        traj_3d_ego = self.traj_3d_ego[idx, :]
        boxes_2d = self.boxes_2d[idx, :]
        kk = self.kk[idx, :]

        
        return inputs, outputs, names, kps, ego_pose, camera_pose, traj_3d_ego, boxes_2d, kk