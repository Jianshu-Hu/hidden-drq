#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=averaged_embedding+contra_loss_weight_002_hidden_512
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=10 weight=0.02 hidden_dim=512 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
