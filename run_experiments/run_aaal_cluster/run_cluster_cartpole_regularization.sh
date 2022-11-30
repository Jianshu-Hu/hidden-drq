#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=regularization
seed=1

echo "start running $tag with seed $seed"
python train.py CBAM=false regularization=2 env=cartpole_swingup batch_size=512 action_repeat=8 num_train_steps=100000 tag=$tag seed=$seed
