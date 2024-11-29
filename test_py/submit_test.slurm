#!/bin/bash
#SBATCH -J test-dit-with1gpu
#SBATCH -p kshdtest
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH -o %x.o%j       #作业标准输出日志，可以自定义
#SBATCH -e %x.e%j       #作业错误输出日志，可以自定义


source ~/miniconda3/etc/profile.d/conda.sh  #激活conda命令
conda activate DiT
#export HF_ENDPOINT="https://hf-mirror.com"  #必须加，否则会出错
export HF_ENDPOINT="https://gh-proxy.com/"

python test.py 

