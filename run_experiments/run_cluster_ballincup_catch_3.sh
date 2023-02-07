#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=rad
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=1 RAD=true aug_when_act=false env=ball_in_cup_catch batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
