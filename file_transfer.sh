source=/bigdata/users/jhu/hidden-drq/outputs/
target=~/PycharmProjects/hidden-drq/outputs/

source2=/bigdata/users/jhu/hidden-drq/runs/
target2=~/PycharmProjects/hidden-drq/runs/

file_name=*
file_name2=*

scp jhu@aaal.ji.sjtu.edu.cn:$source$file_name $target
scp -r jhu@aaal.ji.sjtu.edu.cn:$source2$file_name2 $target2