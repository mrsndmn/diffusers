import torch
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm


dataset_path = "./audio_mnist_continuouse_noise_codes"

noisy_codes_dataset = datasets.load_from_disk(dataset_path)

NUM_VECTORS_IN_CODEBOOK = 1024
NUM_TRAIN_TIMESTEPS = 100

Q_transitioning = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])

for item in tqdm(noisy_codes_dataset):
    noisy_audio_codes = item['noisy_audio_codes']
    clean_audio_codes = item['clean_audio_codes']
    timestep = item['timesteps']

    Q_transitioning[timestep, noisy_audio_codes, clean_audio_codes] += 1


torch.save(Q_transitioning, 'Q_transitioning_raw.pth')