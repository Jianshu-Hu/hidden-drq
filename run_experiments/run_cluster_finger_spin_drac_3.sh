#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=finger_spin+crop+DrAC+trainable_dist
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=7 DrAC=true visualize=true env=finger_spin action_repeat=2 num_train_steps=125000 tag=$tag seed=$seed
