#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=hopper_hop+crop+DrQ_avg_target+01_kl+01_tangent
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 tan_prop=true add_kl_loss=true visualize=true env=hopper_hop action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
