#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=add_hidden_layer_regularization
seed=1

echo "start running $tag with seed $seed"
python train.py regularization=2 env=cheetah_run batch_size=512 action_repeat=4 tag=$tag seed=$seed
