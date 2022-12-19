#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq_new_last_conv_stride_2+log_q_regularized_similarity_init_weight_1
seed=1

echo "start running $tag with seed $seed"
python train.py regularization=8 init_weight=1.0 hidden_dim=1024 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
