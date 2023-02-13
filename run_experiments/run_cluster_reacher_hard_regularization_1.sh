#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=DrQ+t_sne+crop
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=1 aug_when_act=false degrees=180.0 visualize=true env=reacher_hard batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
