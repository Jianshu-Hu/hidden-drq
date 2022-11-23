#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+contra_loss_auto_init_weight_1_target_loss_10_hidden_256
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=7 init_weight=1.0 target_rl=10.0 hidden_dim=256 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
