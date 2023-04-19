#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=quadruped_run+crop+DrAC
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=1 DrAC=true visualize=true env=quadruped_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
