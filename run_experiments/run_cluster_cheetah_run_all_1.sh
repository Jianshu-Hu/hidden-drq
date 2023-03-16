#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+DrQ_avg_target+08_beta_dist+05_critic_tangent+05_kl
seed=1

echo "start running $tag with seed $seed"
python train.py data_aug=5 dist_alpha=0.8 tan_prop=true tan_prop_weight=0.5 add_kl_loss=true init_beta=0.5 visualize=true env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
