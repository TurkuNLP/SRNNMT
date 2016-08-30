#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH -J fg1
#SBATCH -o fg1.out
#SBATCH -e fg1.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

source $USERAPPL/activate_gpu.sh
# run your script
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py
deactivate
