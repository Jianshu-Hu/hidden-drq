#!/bin/bash

cd /home/JI/ji021370910034/workspace/hidden-drq
source /home/JI/ji021370910034/anaconda3/bin/activate
conda activate drq

tag=drq_hidden_256
seed=1

echo "start running $tag with seed $seed"
python train.py regularization=1 hidden_dim=256 env=walker_walk batch_size=512 action_repeat=2 num_train_steps=250000 tag=$tag seed=$seed
