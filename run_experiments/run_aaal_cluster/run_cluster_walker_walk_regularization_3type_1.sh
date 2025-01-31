#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+q_regularization_22_max_update_slower
seed=1

echo "start running $tag with seed $seed"
python train.py regularization=22 init_weight=1.0 env=walker_walk batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
