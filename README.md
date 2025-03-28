<div align="center">
<h1> Unified Human Localization and Trajectory Prediction
with Monocular Vision </h1>
<h3> Po-Chien Luan*, Yang Gao*, Celine Demonsant and Alexandre Alahi
</h3> 
<h4>IEEE International Conference on Robotics & Automation (ICRA) 2025


[[Paper](https://arxiv.org/abs/2503.03535)]
</h4>
<image src="docs/monotransmotion.png" width="350">
</div>

# Getting Started

Create a new conda environment:

```bash
conda create -n mt python=3.9
conda activate mt
```
Install PyTorch from [here](https://pytorch.org/get-started/previous-versions/) accoding to your environment.

Install the requirements using `pip`:
```
pip install -r requirements.txt
```
Install PyTorch3D from [here](https://anaconda.org/pytorch3d/pytorch3d/files) according to your environment. Take Python=3.9, CUDA=11.8, PyTorch=2.4.1 as an example:
```
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
```
# Data
Download the preprocessing data from our release and unzip the data. 
We will release the preprocessing code soon.
# Training and Testing
Train pedestrian localization:
```
python train.py --train_mode loc 
```
Freeze pedestrian localization and train trajectory prediction:
```
python train.py --train_mode freeze_loc
```
Train two modules jointly: 
```
python train.py --train_mode joint
```
Evaluate localization:
```
python eval.py --eval_mode loc 
```
Evaluate trajectory prediction:
```
python eval.py --eval_mode traj_pred 
```
You need to set the following arguments correctly:
- **joints folder**: path to your data.
- **load_loc**: path to your localization checkpoint.
- **load_traj**: path to your trajectory prediction checkpoint.
- **loc_cfg**: path to your configuration of localization model.
- **traj_cfg**: path to your configuration of trajectory prediction model.

Please check `test_script.sh` for some examples.
We also provide checkpoints in release to reproduce the main results in the paper.
# For citation
This repository is work-in-progress and will continue to get updated and improved over the coming months.
```
@article{luan2025unified,
  title={Unified Human Localization and Trajectory Prediction with Monocular Vision},
  author={Luan, Po-Chien and Gao, Yang and Demonsant, Celine and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2503.03535},
  year={2025}
}
```
This work is a follow-up work of [MonoLoco](https://github.com/vita-epfl/monoloco) and [Social-Transmotion](https://github.com/vita-epfl/social-transmotion).
```
@InProceedings{saadatnejad2024socialtransmotion,
    title={Social-Transmotion: Promptable Human Trajectory Prediction}, 
    author={Saadatnejad, Saeed and Gao, Yang and Messaoud, Kaouther and Alahi, Alexandre},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```
```
@InProceedings{bertoni_2019_iccv,
    author = {Bertoni, Lorenzo and Kreiss, Sven and Alahi, Alexandre},
    title = {MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation},
    booktitle = {the IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```