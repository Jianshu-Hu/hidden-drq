#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+lr_1e-3+DrQ_avg_target+actor_two_aug_loss+save_kl
seed=3

echo "start running $tag with seed $seed"
python train.py lr=0.001 data_aug=1 RAD=false avg_target=true add_actor_obs_aug_loss=true visualize=true env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
