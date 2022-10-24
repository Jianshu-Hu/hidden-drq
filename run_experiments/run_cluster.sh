#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=add_hidden_layer_regularization
seed=1

echo "start running $tag with seed $seed"
python train.py env=cartpole_swingup batch_size=512 action_repeat=8 regularization=True
