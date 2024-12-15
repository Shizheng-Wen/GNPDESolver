#!/bin/bash
#SBATCH -n 12                             # ask for cores
#SBATCH --time=30:00:00                   # running time
#SBATCH --mem-per-cpu=8000                # memory for every core
#SBATCH --mail-type=END,FAIL              # 当作业结束或失败时发送邮件
#SBATCH --mail-user=shiwen@student.ethz.ch  # 将此替换为您的邮箱地址
#SBATCH --gpus=rtx_4090:1                   # 请求1个GPU

# 切换到执行目录
cd /cluster/work/math/shiwen/GNPDESolver

# 激活虚拟环境
source ~/venvs/neuralop/bin/activate

echo "Job Name: $SLURM_JOB_NAME"
echo "Config File: $CONFIG_FILE"

# 运行Python脚本
python main.py -c config/$CONFIG_FILE