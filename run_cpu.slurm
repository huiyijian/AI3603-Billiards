#!/bin/bash

#SBATCH --job-name=cpu_run
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH --mail-type=end
#SBATCH --mail-user=lzc050419@sjtu.edu.cn
#SBATCH --qos=high

python ./evaluate.py