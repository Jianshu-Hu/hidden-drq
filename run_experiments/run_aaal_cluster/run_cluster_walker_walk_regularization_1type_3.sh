#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq_smaller_error_target_larger_error_critic
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=32 init_weight=1.0 env=walker_walk batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
