#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+lr_1e-3+DrQ_avg_target+new_beta_kl_target_002+save_kl
seed=2

echo "start running $tag with seed $seed"
python train.py data_aug=1 lr=0.001 RAD=false avg_target=true add_kl_loss=true update_beta=true visualize=true env=cheetah_run action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
