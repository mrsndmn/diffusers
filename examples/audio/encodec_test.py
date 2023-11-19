
import os
import sys
import torchaudio

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')


import datasets
from datasets import Audio
from transformers import EncodecModel, AutoProcessor, DefaultDataCollator
from diffusers import UNet1DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMAudioCodesPipeline

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import sys

from overfit import process_audio_encodec

def test_encodec_shapes():

    encodec_model_name = "facebook/encodec_24khz"

    encodec_model = EncodecModel.from_pretrained(encodec_model_name)
    encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)

    audio_mnist_dataset = datasets.load_from_disk("./audio_mnist")

    audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=24000))

    print("encodec num_quantizers", encodec_model.quantizer.num_quantizers)

    def process_audio(examples):
        return process_audio_encodec(encodec_processor, encodec_model, examples)

    audio_mnist_dataset_24khz_processed = audio_mnist_dataset_24khz.select(range(10))
    audio_mnist_dataset_24khz_processed.set_transform(process_audio)

    collator = DefaultDataCollator()

    train_dataloader = torch.utils.data.DataLoader(audio_mnist_dataset_24khz_processed, collate_fn=collator, batch_size=10, shuffle=False)

    batch = next(iter(train_dataloader))
    print(batch['audio_codes'].shape)


# def test_evaluate():



