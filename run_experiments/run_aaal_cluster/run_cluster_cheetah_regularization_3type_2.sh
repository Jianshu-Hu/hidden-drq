#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+q_regularized_similarity_type_3+without_scheduler
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=6 init_weight=1.0 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
