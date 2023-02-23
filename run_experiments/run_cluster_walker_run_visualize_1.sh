#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+DrQ_remove_01_00_crop+aug_when_act+visualize+deterministic
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=1 RAD=false visualize=true aug_when_act=true env=walker_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
