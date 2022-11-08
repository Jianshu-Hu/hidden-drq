#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=simclr_autoweight
seed=1

echo "start running $tag with seed $seed"
python train.py regularization=6 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
