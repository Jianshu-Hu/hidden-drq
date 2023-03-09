#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+crop+DrQ_not_average_target_div_2+actor_two+visualize+deterministic
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=1 avg_target=false add_actor_obs_aug_loss=true visualize=true env=walker_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
