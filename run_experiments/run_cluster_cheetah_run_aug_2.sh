#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+DrQ_crop+add_actor_obs_aug_loss+aug_when_act+visualize+determinitic
seed=2

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_actor_obs_aug_loss=true aug_when_act=true visualize=true env=cheetah_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
