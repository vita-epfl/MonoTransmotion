import argparse
import json
import os

def get_train_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # general arguments
    parser.add_argument('--joints_folder', type=str, help='Folders storing json file with input joints for training', default="./data/nusc_ped_data/")
    parser.add_argument("--train_mode", type=str, default="", help="Training mode: joint, freeze_loc, loc, traj_pred")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--r_seed", type=int, default=1, help="Random seed")
    parser.add_argument('--no_save', help='to not save model and log file', action='store_true')
    parser.add_argument('--save_every', type=int, help='save model every x epochs')
    parser.add_argument('--eval_every', type=int, default=1, help='evaluate model every x epochs')
    parser.add_argument('--output_dir', type=str, help='output directory', default="./output")
    parser.add_argument('--exp_name', type=str, help='experiment name', default="")
    
    # localization
    parser.add_argument('--dir_loss', help='whether to use direction loss for localization', action='store_true')
    parser.add_argument('--dir_loss_w', type=float, help='weight for direction loss', default=1.0)
    parser.add_argument("--loc_cfg", type=str, default="./configs/localization.yaml", help="Config name for localization")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--epochs_loc", type=int, default=1000, help="Number of epochs for localization")


    # trajectory prediction
    parser.add_argument('--obs', type=int, help='length of observation', default=4)
    parser.add_argument('--pred', type=int, help='length of trajectory prediction', default=6)
    parser.add_argument('--traj_cfg', type=str, default="./configs/traj_pred.yaml", help='config name for trajectory prediction')
    parser.add_argument("--epochs_traj", type=int, default=50, help="Number of epochs for trajectory prediction")
    parser.add_argument("--val_gt", type=bool, default=True, help="Whether to use ground truth for validation")

    # load model
    parser.add_argument("--load_loc", type=str, default="", help="Path to load localization model")
    parser.add_argument("--load_traj", type=str, default="", help="Path to load trajectory prediction model")

    # joint training
    parser.add_argument("--w_loc", type=float, default=0.0, help="Weight for localization loss")
    parser.add_argument("--w_traj", type=float, default=1.0, help="Weight for trajectory prediction loss")

    args = parser.parse_args()
    return args


def get_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--r_seed', type=int, help='Random seed', default=1)
    parser.add_argument('--joints_folder', type=str, help='Folders storing json file with input joints for testing', default="./data/nusc_ped_data/")
    parser.add_argument('--eval_mode', type=str, help='Evaluation mode: loc, traj_pred', default="loc")
    parser.add_argument('--loc_cfg', type=str, help='Config name for localization', default="./configs/localization.yaml")
    parser.add_argument('--traj_cfg', type=str, help='Config name for trajectory prediction', default="./configs/traj_pred.yaml")
    parser.add_argument('--load_loc', type=str, help='Path to load localization model', default="")
    parser.add_argument('--load_traj', type=str, help='Path to load trajectory prediction model', default="")
    parser.add_argument('--obs', type=int, help='length of observation', default=4)
    parser.add_argument('--pred', type=int, help='length of trajectory prediction', default=6)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)

    args = parser.parse_args()
    return args