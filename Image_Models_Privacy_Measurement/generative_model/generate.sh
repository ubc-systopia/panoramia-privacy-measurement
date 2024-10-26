#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=3
#SBATCH --time=01:30:00
#SBATCH --mail-type=ALL

nvidia-smi
cd stylegan2-ada-pytorch/
module purge
module load python/3.7.9 scipy-stack cuda cudnn
source ~/py37/bin/activate

for class in $(seq 0 9); do
  python generate.py --network ../network-snapshot-004233.pkl --outdir samples/ --seeds $(seq 1 1500) --class $class
done
