#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+crop+DrQ_avg_target+05_kl
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true init_beta=0.5 visualize=true env=walker_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
