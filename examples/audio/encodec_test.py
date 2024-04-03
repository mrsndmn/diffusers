
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

from overfit_vq_diffusion import process_audio_encodec

def test_encodec_shapes():

    encodec_model_name = "facebook/encodec_24khz"

    encodec_model = EncodecModel.from_pretrained(encodec_model_name)
    encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)

    # WARN label в этом датасете сейчас - идентификатор спикера!
    audio_mnist_dataset = datasets.load_from_disk("./audio_mnist_full")
    print("audio_mnist_dataset", audio_mnist_dataset[0])

    audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=24000))

    print("encodec num_quantizers", encodec_model.quantizer.num_quantizers)

    def process_audio(examples):
        return process_audio_encodec(encodec_processor, encodec_model, examples)

    audio_mnist_dataset_24khz_processed = audio_mnist_dataset_24khz.select(range(10))
    audio_mnist_dataset_24khz_processed.set_transform(process_audio)

    sample = audio_mnist_dataset_24khz_processed[0]
    audio_codes = sample['audio_codes'][0]

    print("sample", sample)
    assert 'condition_tokens' in sample, "condition_tokens key is in dataset sample"

    # audio_scales = sample['audio_scales'][0]

    audio_code = audio_codes.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    audio_scales = [ None ] # audio_scales.unsqueeze(0).unsqueeze(1).unsqueeze(2)

    print("audio_codes", audio_code.shape)
    # print("audio_scales", audio_scales.shape)

    padding_mask = None
    audio_values = encodec_model.decode(audio_code, audio_scales, padding_mask, return_dict=True).audio_values
    torchaudio.save(f"ddpm-audio-mnist-128/test_samples/sample.wav", audio_values[0].detach().to('cpu'), sample_rate=24000)

    collator = DefaultDataCollator()

    train_dataloader = torch.utils.data.DataLoader(audio_mnist_dataset_24khz_processed, collate_fn=collator, batch_size=10, shuffle=False)

    batch = next(iter(train_dataloader))
    print(batch['audio_codes'].shape)


# def test_evaluate():



