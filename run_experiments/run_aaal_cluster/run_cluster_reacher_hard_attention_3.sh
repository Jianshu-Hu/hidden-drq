#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=attention_actor_critic_share_obs_spatial_weight_1_aug_after_spatial
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=5 CBAM=true env=reacher_hard batch_size=512 action_repeat=4 num_train_steps=125000 tag=$tag seed=$seed
