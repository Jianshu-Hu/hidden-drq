#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+DrQ_crop+add_beta_kl_loss+visualize+determinitic
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true update_beta=true visualize=true env=cheetah_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
