#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=cheetah_run+crop+DrQ_not_average_target_critic_loss_div_2+05_kl_loss+determinitic
seed=2

echo "start running $tag with seed $seed"
python train.py data_aug=1 avg_target=false add_kl_loss=true env=cheetah_run batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
