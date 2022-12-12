#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+contra_loss_hidden_128
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=7 data_aug=0 hidden_dim=128 env=cartpole_swingup batch_size=512 action_repeat=8 num_train_steps=62500 tag=$tag seed=$seed
