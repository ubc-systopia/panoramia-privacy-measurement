#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:0
#SBATCH --gres=gpu:v100:2

cd $project/stylegan2-ada-pytorch
module purge
module load python/3.7.9 scipy-stack cuda/11
source ~/stylegan/bin/activate

python train.py --outdir=$project/latent_space --data=cifar10 --gpus=2 --cfg=cifar --cond=1
