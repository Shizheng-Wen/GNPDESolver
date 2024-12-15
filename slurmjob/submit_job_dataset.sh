#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH -n 16                             # ask for cores
#SBATCH --time=05:00:00                   # running time
#SBATCH --mem-per-cpu=8000                # memory for every core
#SBATCH --mail-type=END,FAIL              # 当作业结束或失败时发送邮件
#SBATCH --mail-user=shiwen@student.ethz.ch  # 将此替换为您的邮箱地址
#SBATCH --output=gen_data.out
#SBATCH --error=gen_data.err

cd /cluster/work/math/shiwen/GNPDESolver/tests

source ~/venvs/neuralop/bin/activate

python gen_data.py