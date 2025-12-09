#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=RTX3090
#SBATCH -N 1
#SBATCH --mail-type=end
#SBATCH --mail-user=lzc050419@sjtu.edu.cn
#SBATCH --gres=gpu:1

python ./evaluate.py