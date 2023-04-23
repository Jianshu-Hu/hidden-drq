#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=finger_turn_hard+crop+DrAC
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=1 DrAC=true visualize=true env=finger_turn_hard action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
