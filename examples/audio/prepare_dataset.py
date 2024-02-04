import torch
import time
import numpy as np
import sys
import os

import torch
import torch.nn.functional as F

from datetime import datetime
print("torch.cuda.is_available()", torch.cuda.is_available())
print("start", datetime.now())

import time

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

from transformers import EncodecModel, AutoProcessor, CLIPTextModel, AutoTokenizer
import datasets
from datasets import load_dataset, Audio

BANDWIDTH = 3.0
MAX_AUDIO_CODES_LENGTH = 256
SAMPLE_RATE = 24000


@torch.no_grad
def _process_audio_encodec(encodec_processor, encodec_model: EncodecModel, clip_tokenizer, example):
    max_audio_sample_len = SAMPLE_RATE * 2
    result = encodec_processor(
        raw_audio=example['audio']['array'],
        padding='max_length',
        max_length=max_audio_sample_len,
        sampling_rate=encodec_processor.sampling_rate,
        return_tensors="pt"
    )
    encoder_outputs = encodec_model.encode(
        result["input_values"].to(encodec_model.device),
        result["padding_mask"].to(encodec_model.device),
        bandwidth=BANDWIDTH,
    )
    # [ bs, channels, num_quantizers, seq_len ]
    audio_codes = encoder_outputs.audio_codes
    # print("1 audio_codes.shape", audio_codes.shape)
    audio_codes = audio_codes[0].permute(0, 2, 1)
    # print("2 audio_codes.shape", audio_codes.shape)
    audio_codes = audio_codes.reshape(audio_codes.shape[0], -1)
    # print("3 audio_codes.shape", audio_codes.shape)
    audio_codes = audio_codes[:, :MAX_AUDIO_CODES_LENGTH]
    clip_processed = clip_tokenizer(str(example["label"]), padding=False, return_tensors="pt")
    return {
        "audio_codes":      audio_codes,
        "attention_mask":   clip_processed["attention_mask"][0],
        "input_ids":        clip_processed["input_ids"][0],
    }


# datasets.logging.set_verbosity_info()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encodec_model_name = "facebook/encodec_24khz"

encodec_model = EncodecModel.from_pretrained(encodec_model_name).to(device)
encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
print("encodec loaded", datetime.now())

clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def process_audio(examples):
    return _process_audio_encodec(encodec_processor, encodec_model, clip_tokenizer, examples)


# WARN label в этом датасете сейчас - идентификатор спикера!
audio_mnist_dataset: datasets.Dataset = load_dataset("audiofolder", data_dir="AudioMNIST/data", split='train')

# сейчас в label хранится идентификатор спикера - делаем по спикерам фильтрацию
audio_mnist_dataset = audio_mnist_dataset.filter(lambda x: x['label'] < 10)
# audio_mnist_dataset = audio_mnist_dataset.select(range(1024))

audio_mnist_dataset = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

# мы хотим, чтобы лейбл был меткой цифры, которую произносят - делаем преобразование
audio_mnist_dataset = audio_mnist_dataset.map(lambda x: { 'label': int(os.path.basename(x['audio']['path']).split('_', maxsplit=1)[0]) })

print("creating processed dataset", datetime.now())
audio_mnist_dataset = audio_mnist_dataset.map(process_audio)

audio_mnist_dataset = audio_mnist_dataset.remove_columns("audio")

print(audio_mnist_dataset)

audio_mnist_dataset.save_to_disk("./audio_mnist_full_encodec_processed")
