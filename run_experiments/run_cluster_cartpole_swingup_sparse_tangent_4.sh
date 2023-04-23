#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cartpole_swingup_sparse+crop+DrQ_avg_target+01_kl+01_q_tan
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true q_tan_prop=true visualize=true env=cartpole_swingup_sparse action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
