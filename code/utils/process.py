
import json
import os
import yaml
import numpy as np
import torch
import torchvision
from .camera import get_keypoints, pixel_to_camera, to_cartesian, back_correct_angles
from pyquaternion import Quaternion
from torchvision import transforms

BF = 0.54 * 721
z_min = 4
z_max = 60
D_MIN = BF / z_max
D_MAX = BF / z_min
Sx = 7.2  # nuScenes sensor size (mm)
Sy = 5.4  # nuScenes sensor size (mm)


def preprocess_monstereo(keypoints, keypoints_r, kk):
    """
    Combine left and right keypoints in all-vs-all settings
    """
    clusters = []
    inputs_l = preprocess_monoloco(keypoints, kk)
    inputs_r = preprocess_monoloco(keypoints_r, kk)

    inputs = torch.empty((0, 68)).to(inputs_l.device)
    for inp_l in inputs_l.split(1):
        clst = 0
        # inp_l = torch.cat((inp_l, cat[:, idx:idx+1]), dim=1)
        for idx_r, inp_r in enumerate(inputs_r.split(1)):
            # if D_MIN < avg_disparities[idx_r] < D_MAX:  # Check the range of disparities
            inp_r = inputs_r[idx_r, :]
            inp = torch.cat((inp_l, inp_l - inp_r), dim=1)  # (1,68)
            inputs = torch.cat((inputs, inp), dim=0)
            clst += 1
        clusters.append(clst)
    return inputs, clusters


def preprocess_monoloco(keypoints, kk, zero_center=False):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    if zero_center:
        kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    else:
        kps_norm = xy1_all
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    # kps_out = torch.cat((kps_out, keypoints[:, 2, :]), dim=1)
    return kps_out


def load_calibration(calibration, im_size, focal_length=5.7):
    if calibration == 'custom':
        kk = [
            [im_size[0]*focal_length/Sx, 0., im_size[0]/2],
            [0., im_size[1]*focal_length/Sy, im_size[1]/2],
            [0., 0., 1.]
        ]
    else:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'intrinsics.yaml')) as a:
            configs = yaml.safe_load(a)
        kk = configs[calibration]['intrinsics']
        orig_size = configs[calibration]['im_size']
        scale = [size / orig for size, orig in zip(im_size, orig_size)]
        kk[0] = [el * scale[0] for el in kk[0]]
        kk[1] = [el * scale[1] for el in kk[1]]
    logger.info("Using {} calibration matrix".format(calibration))
    return kk


def factory_for_gt(path_gt, name=None):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    assert os.path.exists(path_gt), "Ground-truth file not found"
    with open(path_gt, 'r') as f:
        dic_names = json.load(f)
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]

    return dic_gt, kk


def laplace_sampling(outputs, n_samples):

    torch.manual_seed(1)
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda
    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(loc):
    """
    Unnormalize relative bi of a nunmpy array
    Input --> tensor of (m, 2)
    """
    assert loc.size()[1] == 2, "size of the output tensor should be (m, 2)"
    bi = torch.exp(loc[:, 1:2]) * loc[:, 0:1]

    return bi


def preprocess_mask(dir_ann, basename, mode='left'):

    dir_ann = os.path.join(os.path.split(dir_ann)[0], 'mask')
    if mode == 'left':
        path_ann = os.path.join(dir_ann, basename + '.json')
    elif mode == 'right':
        path_ann = os.path.join(dir_ann + '_right', basename + '.json')

    dic = open_annotations(path_ann)
    if isinstance(dic, list):
        return [], []

    keypoints = []
    for kps in dic['keypoints']:
        kps = prepare_pif_kps(np.array(kps).reshape(51,).tolist())
        keypoints.append(kps)
    return dic['boxes'], keypoints


def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True, min_conf=0.):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


def image_transform(image):

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)


def extract_outputs(outputs, tasks=()):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    # print(type(outputs))
    #breakpoint()
    dic_out = {'x': outputs[:, 0:1], 
               'y': outputs[:, 1:2], 
               'd': outputs[:, 2:4], 
               'h': outputs[:, 4:5],
               'w': outputs[:, 5:6],
               'l': outputs[:, 6:7],
               'ori': outputs[:, 7:9]}

    if outputs.shape[1] == 10:
        dic_out['aux'] = outputs[:, 9:10]

    # Multi-task training
    if len(tasks) >= 1:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]
        
    # Preprocess the tensor
    # AV_H, AV_W, AV_L, HWL_STD = 1.72, 0.75, 0.68, 0.1
    bi = unnormalize_bi(dic_out['d'])
    dic_out['bi'] = bi

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    x = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='x')
    y = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='y')
    d = dic_out['d'][:, 0:1]
    z_2 = d**2 - x**2 - y**2
    z_2.clamp_(min=0)

    z = torch.sqrt(z_2) 

    # chck nan
    if torch.isnan(z).any() or torch.isnan(x).any() or torch.isnan(y).any():
        print("z, x, y is nan")
        # for i in range(z.size(0)):
        #     if torch.isnan(z[i]):
        #         print(f"z[{i}] = {z[i]}")
        breakpoint()

    dic_out['xyzd'] = torch.cat((x, y, z, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])
    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry

    if outputs.shape[1] == 10:
        dic_out['aux'] = torch.sigmoid(dic_out['aux'])
    return dic_out


def extract_outputs_no_detach(outputs):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'x': outputs[:, 0:1],
               'y': outputs[:, 1:2],
               'd': outputs[:, 2:4]}

    

    # Preprocess the tensor
    # AV_H, AV_W, AV_L, HWL_STD = 1.72, 0.75, 0.68, 0.1
    dic_out = {key: el for key, el in dic_out.items()}
    x = to_cartesian(outputs[:, 0:3], mode='x')
    y = to_cartesian(outputs[:, 0:3], mode='y')

    d = dic_out['d'][:, 0:1]
    z_2 = d**2 - x**2 - y**2 


    z_2.clamp_(min=0)


    z = torch.sqrt(z_2) 
    # chck nan
    if torch.isnan(z).any() or torch.isnan(x).any() or torch.isnan(y).any():
        breakpoint()
    dic_out['xyzd'] = torch.cat((x, y, z, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    return dic_out


def extract_labels_aux(labels, tasks=None):

    dic_gt_out = {'aux': labels[:, 0:1]}

    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_gt_out[task] for task in tasks]

    dic_gt_out = {key: el.detach().cpu() for key, el in dic_gt_out.items()}
    return dic_gt_out


def extract_labels(labels, tasks=None):

    dic_gt_out = {'x': labels[:, 0:1], 'y': labels[:, 1:2], 'z': labels[:, 2:3], 'd': labels[:, 3:4],
                  'h': labels[:, 4:5], 'w': labels[:, 5:6], 'l': labels[:, 6:7],
                  'ori': labels[:, 7:9], 'aux': labels[:, 10:11]}

    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_gt_out[task] for task in tasks]

    dic_gt_out = {key: el.detach().cpu() for key, el in dic_gt_out.items()}
    return dic_gt_out


def cluster_outputs(outputs, clusters):
    """Cluster the outputs based on the number of right keypoints"""

    # Check for "no right keypoints" condition
    if clusters == 0:
        clusters = max(1, round(outputs.shape[0] / 2))

    assert outputs.shape[0] % clusters == 0, "Unexpected number of inputs"
    outputs = outputs.view(-1, clusters, outputs.shape[1])
    return outputs


def filter_outputs(outputs):
    """Extract a single output for each left keypoint"""

    # Max of auxiliary task
    val = outputs[:, :, -1]
    best_val, _ = val.max(dim=1, keepdim=True)
    mask = val >= best_val
    output = outputs[mask]  # broadcasting happens only if 3rd dim not present
    return output, mask


def extract_outputs_mono(outputs, tasks=None):
    """
    Extract the outputs for single di
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'xyz': outputs[:, 0:3], 'zb': outputs[:, 2:4],
               'h': outputs[:, 4:5], 'w': outputs[:, 5:6], 'l': outputs[:, 6:7], 'ori': outputs[:, 7:9]}

    # Multi-task training
    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    bi = unnormalize_bi(dic_out['zb'])

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    dd = torch.norm(dic_out['xyz'], p=2, dim=1).view(-1, 1)
    dic_out['xyzd'] = torch.cat((dic_out['xyz'], dd), dim=1)

    dic_out['d'], dic_out['bi'] = dd, bi

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])

    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry
    return dic_out


def extract_outputs_no_detach(outputs):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'x': outputs[:, 0:1],
               'y': outputs[:, 1:2],
               'd': outputs[:, 2:4]}

    

    # Preprocess the tensor
    # AV_H, AV_W, AV_L, HWL_STD = 1.72, 0.75, 0.68, 0.1
    dic_out = {key: el for key, el in dic_out.items()}
    x = to_cartesian(outputs[:, 0:3], mode='x')
    y = to_cartesian(outputs[:, 0:3], mode='y')

    d = dic_out['d'][:, 0:1]
    z_2 = d**2 - x**2 - y**2 


    z = torch.sqrt(z_2) 

    dic_out['xyzd'] = torch.cat((x, y, z_2, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    return dic_out


def recover_traj(outputs, ego_pose, camera_pose):

    def rotate(x, quaternion):
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        x_rotate = np.dot(quaternion.rotation_matrix, x)
        return x_rotate
    

    outputs_calib = extract_outputs(outputs)
    # get 3D estimated trajectory 
    xyzd = outputs_calib['xyzd'] # batch_size * seq_length, 4
     

    # calibrate the trajectory
    count = 0
    seq_len = ego_pose.size(1)
    traj_estimated_ls = []

    for i in range(0, xyzd.size(0), seq_len): # 10
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
        traj_estimated_ls.append(traj_3d_ego_i_estimated)

        count += 1

    traj_estimated_ls = np.array(traj_estimated_ls)

    return traj_estimated_ls


def loc2traj(traj_estimated_ls):
    traj_estimated_ls = traj_estimated_ls[:,:,:2]
    # padding zeros: (batch, 10, 2) => (batch, 10, 4)
    traj_estimated_ls = np.concatenate((traj_estimated_ls, np.zeros((traj_estimated_ls.shape[0], traj_estimated_ls.shape[1], 2))), axis=2)
    # (batch, 10, 4) => (batch, 10, 1, 4)
    traj_estimated_ls = np.expand_dims(traj_estimated_ls, axis=2) 
    scene_train_real = traj_estimated_ls.reshape(traj_estimated_ls.shape[0],traj_estimated_ls.shape[1],traj_estimated_ls.shape[2],-1,4) ##(20, n, 16, 4)  # (16, 10, 1, 1, 4)
    scene_train_real_ped = np.transpose(scene_train_real,(0,2,1,3,4)) ## (n, 20, 16, 3) (num_ped (0=ego), frames, keypoints(traj=1), 4) # (16, 1, 10, 1, 4)
    scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])

    num_people_list = []
    for i in range(scene_train_real_ped.shape[0]):
        num_people_list.append(torch.zeros(scene_train_real_ped[i].shape[2]))
    padding_mask = torch.nn.utils.rnn.pad_sequence(num_people_list, batch_first=True, padding_value=1).bool()
    # forward
    scene_train_real_ped = torch.from_numpy(scene_train_real_ped).float()
    scene_train_mask = torch.from_numpy(scene_train_mask).float()

    return scene_train_real_ped, scene_train_mask, padding_mask


def batch_process_coords(coords, masks, padding_mask, config, modality_selection='traj+all', training=False, multiperson=True):
    joints = coords.to(config["DEVICE"])
    masks = masks.to(config["DEVICE"])
    in_F = config["TRAIN"]["input_track_size"]
   
    in_joints_pelvis = joints[:,:, (in_F-1):in_F, 0:1, :].clone()
    in_joints_pelvis_last = joints[:,:, (in_F-2):(in_F-1), 0:1, :].clone()

    joints[:,:,:,0] = joints[:,:,:,0] - joints[:,0:1, (in_F-1):in_F, 0]
    joints[:,:,:,1:] = joints[:,:,:,1:] - joints[:,:,(in_F-1):in_F,1:]

    B, N, F, J, K = joints.shape 
    if training:
        # augment JRDB traj
        joints[:,:,:,0,:3] = getRandomRotatePoseTransform(config)(joints[:,:,:,0,:3])

    joints = joints.transpose(1, 2).reshape(B, F, N*J, K)
    in_joints_pelvis = in_joints_pelvis.reshape(B, 1, N, K)
    in_joints_pelvis_last = in_joints_pelvis_last.reshape(B, 1, N, K)
    masks = masks.transpose(1, 2).reshape(B, F, N*J)
    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]   
    in_joints = joints[:,:in_F].float()
    out_joints = joints[:,in_F:in_F+out_F].float()
    in_masks = masks[:,:in_F].float()
    out_masks = masks[:,in_F:in_F+out_F].float()

 
    return in_joints, in_masks, out_joints, out_masks, padding_mask.float(), in_joints_pelvis


def getRandomRotatePoseTransform(config):
    """
    Performs a random rotation about the origin (0, 0, 0)
    """

    def do_rotate(pose_seq):
        B, F, J, K = pose_seq.shape

        angles = torch.deg2rad(torch.rand(B)*360)

        rotation_matrix = torch.zeros(B, 3, 3).to(pose_seq.device)

        ## rotate around z axis (vertical axis)
        rotation_matrix[:,0,0] = torch.cos(angles)
        rotation_matrix[:,0,1] = -torch.sin(angles)
        rotation_matrix[:,1,0] = torch.sin(angles)
        rotation_matrix[:,1,1] = torch.cos(angles)
        rotation_matrix[:,2,2] = 1

        rot_pose = torch.bmm(pose_seq.reshape(B, -1, 3).float(), rotation_matrix)
        rot_pose = rot_pose.reshape(pose_seq.shape)
        return rot_pose

    return transforms.Lambda(lambda x: do_rotate(x))