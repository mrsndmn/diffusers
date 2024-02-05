#!/bin/env bash
#SBATCH --job-name="pepare_dataset"
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

set -eo pipefail
set -o xtrace

#Executable

export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/huggingface_hub/src:/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src:/home/dtarasov/workspace/hse-audio-dalle2/transformers/src:/home/dtarasov/workspace/hse-audio-dalle2/frechet-audio-distance

srun /home/dtarasov/.conda/envs/discrete_diffusion_diffusers/bin/python3 prepare_dataset.py
# srun /home/dtarasov/.conda/envs/discrete_diffusion_diffusers/bin/accelerate launch --config_file accelerate_configs/single_gpu_config.yaml audio_dalle_full_inference.py $@