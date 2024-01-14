#!/bin/env bash
#SBATCH --job-name="dummy_q_inf"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=2       # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus-per-node=1       # Количество GPU на один узел
#SBATCH -C type_a|type_b|type_c|type_e
#SBATCH --time-min=60

module purge
module load Python
module load CUDA/10.2
module list

source /opt/software/python/anaconda/2021_11/bin/deactivate
source /opt/software/python/anaconda/2021_11/bin/activate audio_dalle2
nvidia-smi

set -eo pipefail
set -o xtrace

#Executable

export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/huggingface_hub/src:/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src:/home/dtarasov/workspace/hse-audio-dalle2/transformers/src:/home/dtarasov/workspace/hse-audio-dalle2/frechet-audio-distance

srun /home/dtarasov/.conda/envs/audio_dalle2/bin/python3 official_dummy_inference.py $@
# srun /home/dtarasov/.conda/envs/audio_dalle2/bin/accelerate launch --config_file accelerate_configs/single_gpu_config.yaml audio_dalle_full_inference.py $@