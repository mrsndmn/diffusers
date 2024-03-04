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
MAX_AUDIO_CODES_LENGTH = 2560
SAMPLE_RATE = 24000
MAX_AUDIO_SAMPLE_LEN = SAMPLE_RATE * 5

csv_data_dir = '~/workspace/hse-audio-dalle2/audiocaps/dataset/'
audio_data_dir = '/home/dtarasov/workspace/hse-audio-dalle2/data/audiocaps_train_normalized/'
processed_data_dir = '/home/dtarasov/workspace/hse-audio-dalle2/data/audiocaps_train_processed/'

@torch.no_grad
def _process_audio_encodec(encodec_processor, encodec_model: EncodecModel, clip_tokenizer, example):
    result = encodec_processor(
        raw_audio=example['audio']['array'],
        padding='max_length',
        max_length=MAX_AUDIO_SAMPLE_LEN,
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
    print("3 audio_codes.shape", audio_codes.shape)
    audio_codes = audio_codes[:, :MAX_AUDIO_CODES_LENGTH]
    print("4 audio_codes.shape", audio_codes.shape)
    clip_processed = clip_tokenizer(str(example["caption"]), padding=False, return_tensors="pt")
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
audiocaps_dataset: datasets.Dataset = datasets.load_dataset('csv', data_dir=csv_data_dir)
audiocaps_dataset = audiocaps_dataset['train']
# audiocaps_dataset = audiocaps_dataset.select(range(10))

audio_files = set(os.listdir(audio_data_dir))

audiocaps_dataset = audiocaps_dataset.filter(lambda x: x['youtube_id'] + ".wav" in audio_files)
audiocaps_dataset = audiocaps_dataset.map(lambda x: { 'audio':  audio_data_dir +  x['youtube_id'] + '.wav'})

# сейчас в label хранится идентификатор спикера - делаем по спикерам фильтрацию

audiocaps_dataset = audiocaps_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

print("creating processed dataset", datetime.now())
audiocaps_dataset = audiocaps_dataset.map(process_audio)

audiocaps_dataset = audiocaps_dataset.remove_columns("audio")

print("audiocaps_dataset", audiocaps_dataset)

print("going to save dataset", datetime.now())
# audiocaps_dataset.save_to_disk("./audio_mnist_full_encodec_processed") # only 10 speakers
audiocaps_dataset.save_to_disk(processed_data_dir)
print("done")