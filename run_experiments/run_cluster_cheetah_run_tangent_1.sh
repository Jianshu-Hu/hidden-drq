#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+SAC+weight_01_tangent_prop_variance+visualize+determinitic
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=-1 tangent_prop=true RAD=true visualize=true env=cheetah_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
