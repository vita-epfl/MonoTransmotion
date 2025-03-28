
# Localization
python train.py --train_mode loc --bs 512 --dir_loss --dir_loss_w 0.05


# Trajectory prediction
python train.py --train_mode freeze_loc --bs 32 --load_loc '../checkpoints/loc/best_loc_model.pth' 

# Joint train two models
python train.py --train_mode joint --bs 32 --load_loc '../checkpoints/loc/best_loc_model.pth' --load_traj '../checkpoints/traj_pred/best_traj_model.pth' --loc_cfg "./configs/localization_joint.yaml" --traj_cfg "./configs/traj_pred_joint.yaml"


# Evaluation
# Localization
python eval.py --eval_mode loc --load_loc '../checkpoints/loc/best_loc_model.pth' --bs 32
# Trajectory prediction
python eval.py --eval_mode traj_pred --load_traj '../checkpoints/traj_pred/best_traj_model.pth' --load_loc '../checkpoints/loc/best_loc_model.pth' --bs 1
