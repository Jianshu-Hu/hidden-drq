#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+smooth_l1_weight_05_averaged_target_hidden_512
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=9 weight=0.5 hidden_dim=512 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
