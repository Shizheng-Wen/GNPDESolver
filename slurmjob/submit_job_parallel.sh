#!/bin/bash
#SBATCH -n 12                             # ask for cores
#SBATCH --time=30:00:00                   # running time
#SBATCH --mem-per-cpu=8000                # memory for every core
#SBATCH --mail-type=END,FAIL              # 当作业结束或失败时发送邮件
#SBATCH --mail-user=shiwen@student.ethz.ch  # 将此替换为您的邮箱地址
#SBATCH --gpus=rtx_4090:2                   # 请求1个GPU

# 切换到执行目录
cd /cluster/work/math/shiwen/GNPDESolver

# 激活虚拟环境
source ~/venvs/neuralop/bin/activate

echo "Job Name: $SLURM_JOB_NAME"
echo "Config File: $CONFIG_FILE"

# 设置环境变量
export OMP_NUM_THREADS=1

# 运行Python脚本
torchrun --nproc_per_node=2 main.py -c config/$CONFIG_FILE