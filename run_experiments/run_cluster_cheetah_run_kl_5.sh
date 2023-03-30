#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+DrQ_avg_target+01_kl+scheduled_08_beta_dist
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=5 init_dist_alpha=0.8 final_dist_alpha=1.0 add_kl_loss=true init_beta=0.1 visualize=true env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
