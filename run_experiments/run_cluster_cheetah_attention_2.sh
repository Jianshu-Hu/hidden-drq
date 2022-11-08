#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq

tag=attention_regularization_last_layer
seed=2

echo "start running $tag with seed $seed"
python train.py regularization=5 CBAM=true env=cheetah_run batch_size=512 action_repeat=4 num_train_steps=250000 tag=$tag seed=$seed
