#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq_hidden_128
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=1 hidden_dim=128 env=reacher_hard batch_size=512 action_repeat=4 num_train_steps=125000 tag=$tag seed=$seed
