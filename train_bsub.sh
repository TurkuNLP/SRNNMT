#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 34:00:00
#SBATCH -J mt1
#SBATCH -o mt1.out
#SBATCH -e mt1.err
#SBATCH --gres=gpu:1
#SBATCH --mem=160000

source $USERAPPL/activate_gpu.sh
# run your script
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py
deactivate
