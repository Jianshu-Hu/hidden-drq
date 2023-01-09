#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=drq+label_25_cluster_contra_loss+without_scheduler
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=3 n_clusters=25 init_weight=0.1 env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
