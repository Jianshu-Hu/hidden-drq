#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+DrQ_alpha_08_crop+kl_loss+visualize+deterministic
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=5 dist_alpha=0.8 degrees=5.0 add_kl_loss=true visualize=true env=cheetah_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
