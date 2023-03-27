#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+crop+DrQ_avg_target+scheduled_07_beta_dist
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=5 dist_alpha=0.7 RAD=false avg_target=true visualize=true env=walker_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
