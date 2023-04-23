
#!/bin/bash

cd /bigdata/users/jhu/hidden-drq
source /bigdata/users/yjiang/miniconda3/bin/activate
conda activate drq_test

tag=reacher_easy+crop+DrQ_avg_target
seed=4

echo "start running $tag with seed $seed"
python train.py data_aug=1 visualize=true env=reacher_easy action_repeat=2 num_train_steps=125000 tag=$tag seed=$seed
