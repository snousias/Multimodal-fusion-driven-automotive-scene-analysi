#!/bin/bash

DIR=carla_sharing_vq20_sgd_v2

FULLDIR=$(pwd)/$DIR



! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model /home/stavros/Workspace/SqueezeDet-PyTorch/exp/carla_training_adam_v1/model_last.pth --exp_id temp --batch_size 2 --debug 2)
/home/stavros/anaconda3/envs/object-detection-keras/bin/python /home/stavros/Workspace/kitti-object-eval-python/evaluate.py evaluate --label_path=/home/stavros/Desktop/OpenPCDet/data/carla/training/label_2/ --result_path=/home/stavros/Workspace/SqueezeDet-PyTorch/exp/temp/results/data/ --label_split_file=/home/stavros/Desktop/OpenPCDet/data/carla/ImageSets/val.txt --current_class=1 --coco=False



for path in $FULLDIR/*; do

if [[ $path == *"last"* ]]; then

    part1=$(dirname "$path")
    part2=$(basename "$path")
    # echo $part2
    id="$(cut -d'_' -f3 <<<"$part2")"
    echo $path
    evaldir="eval_"$DIR"_layer_"$id
    echo $evaldir

    ! $(/home/stavros/anaconda3/envs/pcdet/bin/python ../src/main.py --mode eval --load_model $path --exp_id temp --batch_size 2 --debug 2)
    /home/stavros/anaconda3/envs/object-detection-keras/bin/python /home/stavros/Workspace/kitti-object-eval-python/evaluate.py evaluate --label_path=/home/stavros/Desktop/OpenPCDet/data/carla/training/label_2/ --result_path=/home/stavros/Workspace/SqueezeDet-PyTorch/exp/temp/results/data/ --label_split_file=/home/stavros/Desktop/OpenPCDet/data/carla/ImageSets/val.txt --current_class=1 --coco=False

fi
    
done

