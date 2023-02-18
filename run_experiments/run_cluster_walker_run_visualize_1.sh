#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+DrQ_rotation_shift+visualize
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=4 RAD=false visualize=true aug_when_act=false env=walker_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
