#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=pendulum_swingup+crop+DrQ_avg_target+01_kl+01_q_tan
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true add_q_tan=true visualize=true env=pendulum_swingup action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
