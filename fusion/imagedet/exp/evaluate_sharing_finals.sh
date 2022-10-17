#!/bin/bash


DIR=carla_training_adam_v1
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/model_last.pth --exp_id temp --batch_size 2 --debug 2)


DIR=carla_sharing_dl10_sgd_v2
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/modelopt_layer_20_last.pth --exp_id eval_$DIR --batch_size 2 --debug 2)

DIR=carla_sharing_dl20_sgd_v2
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/modelopt_layer_20_last.pth --exp_id eval_$DIR --batch_size 2 --debug 2)

DIR=carla_sharing_dl30_sgd_v2
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/modelopt_layer_20_last.pth --exp_id eval_$DIR --batch_size 2 --debug 2)

DIR=carla_sharing_vq10_sgd_v2
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/modelopt_layer_20_last.pth --exp_id eval_$DIR --batch_size 2 --debug 2)

DIR=carla_sharing_vq20_sgd_v2
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/modelopt_layer_20_last.pth --exp_id eval_$DIR --batch_size 2 --debug 2)

DIR=carla_sharing_vq30_sgd_v2
! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/$DIR/modelopt_layer_20_last.pth --exp_id eval_$DIR --batch_size 2 --debug 2)

