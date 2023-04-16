#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=reacher_hard+crop+DrAC_trainable_dist
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=7 DrAC=true visualize=true env=reacher_hard action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
