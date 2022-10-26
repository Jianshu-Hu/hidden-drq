#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=regularization_without_drq
seed=1

echo "start running $tag with seed $seed"
python train.py CBAM=false regularization=2 env=cheetah_run batch_size=512 action_repeat=4 tag=$tag seed=$seed
