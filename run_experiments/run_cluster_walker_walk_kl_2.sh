#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_walk+crop+DrQ_avg_target+01_kl
seed=2

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true visualize=true env=walker_walk action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
