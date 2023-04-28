#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cup_catch+crop+DrAC
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 DrAC=true visualize=true env=ball_in_cup_catch action_repeat=2 num_train_steps=125000 tag=$tag seed=$seed
