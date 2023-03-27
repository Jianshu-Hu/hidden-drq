#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=finger_spin+crop+DrQ_avg_target+01_kl
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=1 RAD=false avg_target=true add_kl_loss=true init_beta=0.1 visualize=true env=finger_spin action_repeat=2 num_train_steps=125000 tag=$tag seed=$seed
