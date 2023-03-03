#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=walker_run+DrQ_croop+add_kl_loss+visualize+deterministic
seed=5

echo "start running $tag with seed $seed"
python train.py data_aug=1 add_kl_loss=true aug_when_act=false visualize=true env=walker_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
