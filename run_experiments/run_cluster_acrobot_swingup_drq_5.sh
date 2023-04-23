#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=acrobot_swingup+crop+DrQ_avg_target
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=1 visualize=true env=acrobot_swingup action_repeat=2 num_train_steps=500000 tag=$tag seed=$seed
