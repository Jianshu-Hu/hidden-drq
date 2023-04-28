#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+crop+DrQ_avg_target+01_kl+01_q_tan_prop+01_a_tan_on_latent
seed=3

echo "start running $tag with seed $seed"
python train.py data_aug=1 a_tan_prop=true q_tan_prop=true add_kl_loss=true visualize=true env=walker_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
