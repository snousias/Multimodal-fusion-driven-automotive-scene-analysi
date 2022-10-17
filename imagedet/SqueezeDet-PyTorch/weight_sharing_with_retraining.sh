#!/usr/bin/env bash

python src/main.py \
--mode sharing \
--load_model exp/carla_training_adam_v1/model_best.pth  \
--exp_id carla_sharing_dl10_v3 \
--batch_size 16 \
--lr 0.001 \
--num_epochs 8 \
--sharing_method dl \
--sharing_acceleration_factor 10


