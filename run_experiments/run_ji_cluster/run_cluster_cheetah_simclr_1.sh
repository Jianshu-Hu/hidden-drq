#!/bin/bash

cd /home/JI/ji021370910034/workspace/hidden-drq
source /home/JI/ji021370910034/anaconda3/bin/activate
conda activate drq

tag=drq+crop+rotation+contra_loss_hidden_256
seed=1

echo "start running $tag with seed $seed"
python train.py regularization=7 data_aug=3 hidden_dim=256 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
