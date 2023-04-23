#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cartpole_swingup_sparse+crop+RAD
seed=2

echo "start running $tag with seed $seed"
python train.py data_aug=1 RAD=true visualize=true env=cartpole_swingup_sparse action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
