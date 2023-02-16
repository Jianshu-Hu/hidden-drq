#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+RAD+aug_when_act+visualize_crop
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=1 RAD=true aug_when_act=true visualize=true env=walker_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
