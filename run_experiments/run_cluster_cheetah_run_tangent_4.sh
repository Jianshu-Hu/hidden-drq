#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+DrQ_avg_target+05_critic_tangent_prop
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 critic_tangent_prop=true critic_tangent_prop_weight=0.5 visualize=true env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
