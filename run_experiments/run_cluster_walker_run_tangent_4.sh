#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+crop+DrQ_avg_target+01_kl+01_q_tan_prop+001_a_tan_prop
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 q_tan_prop=true a_tan_prop=true a_tan_prop_weight=0.01 add_kl_loss=true visualize=true env=walker_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
