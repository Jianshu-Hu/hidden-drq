#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+init_weight_015_simclr_on_second_to_last+simclr_on_last+hidden_256
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=12 init_weight=0.15 hidden_dim=256 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
