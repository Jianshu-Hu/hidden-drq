#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=drq+rotation_15_180+aug_when_act
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=2 degrees=180.0 aug_when_act=true M=1 env=reacher_hard batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
