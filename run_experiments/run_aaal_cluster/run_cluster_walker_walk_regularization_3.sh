#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=l2_regularization_weight_1
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=2 env=walker_walk batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
