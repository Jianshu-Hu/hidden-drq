#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq_new_last_conv_stride_2
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=1 hidden_dim=1024 env=walker_walk batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
