#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+hue_01+crop+contra_loss_hidden_256
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=7 data_aug=2 hidden_dim=256 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
