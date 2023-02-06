#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=drq+rotation_30+aug_when_act
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=2 aug_when_act=true batch_size=512 env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
