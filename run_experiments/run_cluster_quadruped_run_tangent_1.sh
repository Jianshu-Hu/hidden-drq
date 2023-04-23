#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=quadruped_run+crop+DrQ_avg_target+01_kl+01_tangent+train_longer
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true q_tan_prop=true visualize=true env=quadruped_run action_repeat=2 num_train_steps=500000 tag=$tag seed=$seed
