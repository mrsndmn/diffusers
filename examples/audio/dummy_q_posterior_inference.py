import argparse
import sys

import torch
import torch.nn.functional as F

from datetime import datetime
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

from overfit_vq_diffusion import TrainingConfig, NUM_TRAIN_TIMESTEPS, BANDWIDTH, SAMPLE_RATE

from transformers import EncodecModel, AutoProcessor, DefaultDataCollator, CLIPTextModel, AutoTokenizer

from frechet_audio_distance import FrechetAudioDistance

import torchaudio

config = TrainingConfig()

dense_dummy_scheduler = True

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

variant = "q_posterior_official_repo_aux_only_timesteps_importance_sampling_transitioning_matricies_plus_eye2024-01-07 15:36:11.188800"
model = Transformer2DModel.from_pretrained("ddpm-audio-mnist-128/", variant=variant, use_safetensors=True)
assert model.is_input_continuous == False, 'transformer is discrete'


if dense_dummy_scheduler:
    noise_scheduler = VQDiffusionDenseTrainedDummyQPosteriorScheduler(
        q_transition_martices_path=config.noise_scheduler_q_transition_martices_path,
        q_transition_cummulative_martices_path=config.noise_scheduler_q_transition_cummulative_martices_path,
        q_transition_transposed_martices_path=config.noise_scheduler_q_transition_transposed_martices_path,
        q_transition_transposed_cummulative_martices_path=config.noise_scheduler_q_transition_transposed_cummulative_martices_path,
        device=device,
    )
else:
    noise_scheduler = VQDiffusionSchedulerDummyQPosterior(
        num_vec_classes=model.num_vector_embeds,
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        device=device,
    )


encodec_model_name = "facebook/encodec_24khz"

encodec_model = EncodecModel.from_pretrained(encodec_model_name)
encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)

clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

encodec_model = encodec_model.to(device)
encodec_model.eval()

clip_text_model = clip_text_model.to(device)
clip_text_model.eval()

model = model.to(device)
model.eval()

pipeline = VQDiffusionAudioTextConditionalPipeline(
    encodec=encodec_model,
    clip_tokenizer=clip_tokenizer,
    clip_text_model=clip_text_model,
    transformer=model,
    scheduler=noise_scheduler,
)

condition_classes = list(range(10))
text_condition = [ str(x) for x in condition_classes ]
pipeline_out = pipeline(
    num_inference_steps=NUM_TRAIN_TIMESTEPS,
    bandwidth=BANDWIDTH,
    num_generated_audios = 10,
    text_condition=text_condition,
)
audio_codes = pipeline_out.audio_codes
audio_values = pipeline_out.audio_values


test_dir = os.path.join(config.output_dir, config.experiment_name, "samples_dummy_q_posterior")
generated_samples_path = test_dir
generated_samples_path = Path(generated_samples_path)
generated_samples_path.mkdir(parents=True, exist_ok=True)

print("will be saved to", generated_samples_path)

for i in range(audio_values.shape[0]):
    current_text_condition = text_condition[i]
    audio_wave = audio_values[i]
    torchaudio.save(f"{generated_samples_path}/{current_text_condition}.wav", audio_wave.to('cpu'), sample_rate=SAMPLE_RATE)

print("done")

# todo eval

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