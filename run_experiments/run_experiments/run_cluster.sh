#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq_with_attention
seed=1

echo "start running $tag with seed $seed"
python train.py CBAM=true regularization=1 env=cartpole_swingup batch_size=512 action_repeat=8 tag=$tag seed=$seed
