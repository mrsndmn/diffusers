import sys

import torch

from torch.utils.data import DataLoader

from datetime import datetime
print("torch.cuda.is_available()", torch.cuda.is_available())
print("start", datetime.now())
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/huggingface_hub/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

import os
import torchaudio

import datasets
from datasets import Audio, Dataset, load_dataset

from transformers import EncodecModel, AutoProcessor, DefaultDataCollator


import torch

from tqdm.auto import tqdm
import os

NUM_EPOCHS = 2
NUM_VECTORS_IN_CODEBOOK = 1024
MAX_AUDIO_CODES_LENGTH = 256
NUM_TRAIN_TIMESTEPS = 100
SAMPLE_RATE = 24000
BANDWIDTH = 3.0
MAX_AUDIO_SAMPLE_LEN = int(SAMPLE_RATE * 1.5)


encodec_model_name = "facebook/encodec_24khz"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encodec_model = EncodecModel.from_pretrained(encodec_model_name).to(device)
encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
print("encodec loaded", datetime.now())

audio_mnist_dataset: datasets.Dataset = load_dataset("audiofolder", data_dir="AudioMNIST/data", split='train')

audio_mnist_dataset = audio_mnist_dataset.filter(lambda x: x['label'] < 10)
# audio_mnist_dataset = audio_mnist_dataset.select(range(1024))

audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def preprocess_waveforms(batch):
    raw_audios = []
    for item in batch['audio']:
        raw_audios.append(item['array'])
    audios_processed = encodec_processor(
        raw_audio=raw_audios,
        padding='max_length',
        max_length=MAX_AUDIO_SAMPLE_LEN,
        sampling_rate=encodec_processor.sampling_rate,
        return_tensors="pt"
    )
    return audios_processed

audio_mnist_dataset_24khz = audio_mnist_dataset_24khz.map(preprocess_waveforms, batched=True, remove_columns=['audio'])

# print(audio_mnist_dataset_24khz[0])
# audio_mnist_dataset_24khz = audio_mnist_dataset_24khz.select(range(10))

BATCH_SIZE = 1
dataloader = DataLoader(audio_mnist_dataset_24khz, batch_size=BATCH_SIZE, collate_fn=DefaultDataCollator())

# print(next(iter(dataloader)))

all_labels = []
all_timesteps = []
all_audio_codes_noisy = []
all_audio_codes_clean = []

with torch.no_grad():
    #
    timesteps_batch = torch.arange(NUM_TRAIN_TIMESTEPS).to(device)
    timesteps_batch = timesteps_batch.repeat(BATCH_SIZE)
    #
    noise_schedule = torch.linspace(1e-4, 1e-1, steps=NUM_TRAIN_TIMESTEPS).to(device)
    noise_schedule = noise_schedule.unsqueeze(1).unsqueeze(2).repeat(BATCH_SIZE, 1, MAX_AUDIO_SAMPLE_LEN).to(device) # [ bs, timesteps, MAX_AUDIO_SAMPLE_LEN ]
    print("noise_schedule", noise_schedule.shape)
    noise_mean = torch.zeros_like(noise_schedule)
    # run epochs
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(dataloader):
            # for timestep in range(NUM_TRAIN_TIMESTEPS+1):
            #
            input_values = batch["input_values"].to(encodec_model.device)
            input_values = input_values.repeat(NUM_TRAIN_TIMESTEPS, 1, 1)
            noisy_input_values = input_values + torch.normal(mean=noise_mean, std=noise_schedule)
            padding_mask = batch["padding_mask"].to(encodec_model.device)
            padding_mask = padding_mask.repeat(NUM_TRAIN_TIMESTEPS, 1)
            print("noisy_input_values", noisy_input_values.shape)
            batch_labels = batch['labels'].repeat(NUM_TRAIN_TIMESTEPS)
            #
            encoder_outputs = encodec_model.encode(
                noisy_input_values,
                padding_mask,
                bandwidth=BANDWIDTH,
            )
            #
            audio_codes = encoder_outputs.audio_codes
            print("1 audio_codes.shape", audio_codes.shape)
            audio_codes = audio_codes[0].permute(0, 2, 1)   # [ bs, seq_len, num_quantizers ]
            print("2 audio_codes.shape", audio_codes.shape)
            audio_codes = audio_codes.reshape(audio_codes.shape[0], -1) # [ bs, reshaped_seq_len ]
            print("3 audio_codes.shape", audio_codes.shape)
            audio_codes = audio_codes[:, :MAX_AUDIO_CODES_LENGTH] # [ bs, MAX_AUDIO_CODES ]
            print("audio_codes", audio_codes.shape)

            batch_labels = batch_labels.to('cpu')
            timesteps_batch = timesteps_batch.to('cpu')
            audio_codes = audio_codes.to('cpu')

            expected_batch_size = batch_labels.shape[0]

            assert timesteps_batch.shape[0] == expected_batch_size, f'timesteps_batch {timesteps_batch.shape[0]} != {expected_batch_size}'
            assert audio_codes.shape[0] == expected_batch_size, f'audio_codes {audio_codes.shape[0]} != {expected_batch_size}'

            for i in range(expected_batch_size):
                all_labels.append(batch_labels[i])
                all_timesteps.append(timesteps_batch[i])
                all_audio_codes_noisy.append(audio_codes[i])
                all_audio_codes_clean.append(audio_codes[0]) # corresponds to almost clean audio codes


dataset_dict = {
    "labels":            all_labels,
    "timesteps":         all_timesteps,
    "noisy_audio_codes": all_audio_codes_noisy,
    "clean_audio_codes": all_audio_codes_clean,
}

noisy_audio_codes_dataset = Dataset.from_dict(dataset_dict)
noisy_audio_codes_dataset = noisy_audio_codes_dataset.shuffle(seed=42)
noisy_audio_codes_dataset = noisy_audio_codes_dataset.flatten_indices()
noisy_audio_codes_dataset.save_to_disk("./audio_mnist_continuouse_noise_codes")

print("Done.")