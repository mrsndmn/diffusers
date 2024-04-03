#!/bin/env bash
#SBATCH --job-name="classifier"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=2       # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus-per-node=1       # Количество GPU на один узел
#SBATCH -C type_a|type_b|type_c|type_e
#SBATCH --time-min=120

module purge
module load Python
module load CUDA/10.2
module list

source /opt/software/python/anaconda/2021_11/bin/deactivate
source /opt/software/python/anaconda/2021_11/bin/activate discrete_diffusion_diffusers
nvidia-smi

nvcc --version

set -eo pipefail
set -o xtrace

export LD_LIBRARY_PATH=/home/dtarasov/.conda/envs/discrete_diffusion_diffusers/lib:/home/dtarasov/.conda/envs/audio_dalle2/lib/python3.10/site-packages/torch/lib:/home/dtarasov/.conda/envs/audio_dalle2/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export RPATH=/home/dtarasov/.conda/envs/discrete_diffusion_diffusers/lib

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "RPATH=$RPATH"

#Executable

export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/huggingface_hub/src:/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src:/home/dtarasov/workspace/hse-audio-dalle2/transformers/src:/home/dtarasov/workspace/hse-audio-dalle2/frechet-audio-distance

srun /home/dtarasov/.conda/envs/discrete_diffusion_diffusers/bin/python3 ./audio_mnist_classifier.py
