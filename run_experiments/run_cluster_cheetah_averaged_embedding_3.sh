#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=averaged_embedding_for_Q_target_Q+contra_loss_cosine_weight_hidden_256
seed=3

echo "start running $tag with seed $seed"
python train.py regularization=10 init_weight=0.1 hidden_dim=256 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
