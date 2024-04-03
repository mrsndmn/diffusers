import argparse
import sys

import torch
import torch.nn.functional as F

from datetime import datetime

from diffusers.pipelines.pipeline_utils import AudioCodesPipelineOutput
print("torch.cuda.is_available()", torch.cuda.is_available())
print("start", datetime.now())
script_start_time = datetime.now()

import time
import yaml

from pathlib import Path

# todo remove hardcode
if torch.backends.mps.is_available():
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/frechet-audio-distance')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/diffusers/src')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/transformers/src')
else:
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

from diffusers import VQDiffusionScheduler, Transformer2DModel, VQDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers.pipelines.deprecated.vq_diffusion.pipeline_vq_diffusion import VQDiffusionAudioTextConditionalPipeline

from dataclasses import dataclass
import torch

from tqdm.auto import tqdm
import os

from diffusers.schedulers.scheduling_vq_diffusion import VQDiffusionDenseTrainedDummyQPosteriorScheduler, VQDiffusionSchedulerDummyQPosterior

from overfit_vq_diffusion import TrainingConfig, BANDWIDTH, SAMPLE_RATE

from transformers import EncodecModel, AutoProcessor, DefaultDataCollator, CLIPTextModel, AutoTokenizer

from frechet_audio_distance import FrechetAudioDistance

import torchaudio

config = TrainingConfig()

NUM_TRAIN_TIMESTEPS = config.num_train_timesteps

generated_samples_path = 'ddpm-audio-mnist-128/q_posterior_official_repo_aux_only_dummy_q_posterior_reconstruct_30_monitoring/samples/2024-02-05 10:45:21.810084/149/start_timestep_50/'

frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False,
)

fad_score = frechet.score(
    "audio_mnist_full/audios/",
    generated_samples_path,
    background_embds_path="audio_mnist_full/audios/frechet_embeddings.npy",
)

print("fad_score", fad_score)