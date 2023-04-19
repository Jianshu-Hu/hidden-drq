#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+crop+DrAC+trainable_dist_target_error
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=7 DrAC=true visualize=true env=walker_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
