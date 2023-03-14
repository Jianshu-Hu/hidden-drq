#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+DrQ_avg_target+critic_tangent_prop
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 tangent_prop=true visualize=true env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
