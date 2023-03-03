#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=reacher_hard+DrQ_alpha_08_rotation+aug_when_act+visualize+deterministic
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=6 dist_alpha=0.8 degrees=180.0 aug_when_act=true visualize=true env=reacher_hard batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
