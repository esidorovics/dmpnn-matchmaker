#!/bin/bash

#SBATCH --job-name=hyper4
#SBATCH --partition=gpu
#SBATCH --time=120:00:00

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=4

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --tasks-per-node=1


# Load modules or your own conda environment here
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load cuDNN/8.0.3.33-gcccuda-2020a
module load NCCL/2.7.8-gcccuda-2020a
module load Miniconda3/4.7.12.1
source activate thesis

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
  sleep 5
done
##############################################################################################

#### call your code below
python -u hyperparameter_optimization.py --data_path=/home/eduards1/thesis/data/DrugCombData.csv --cell_lines=/home/eduards1/thesis/data/cell_line_gex.csv --datapoints_path=/home/eduards1/thesis/data/datapoints --train_index=/home/eduards1/thesis/data/train_inds.txt --val_index=/home/eduards1/thesis/data/val_inds.txt --test_index=/home/eduards1/thesis/data/test_inds.txt --epochs=1000 --features_generator rdkit_2d_normalized --no_features_scaling --num_iters=150

exit
