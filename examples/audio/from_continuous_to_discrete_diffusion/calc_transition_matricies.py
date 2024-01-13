import torch
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm


dataset_path = "./audio_mnist_continuouse_noise_codes_300_timesteps/"

noisy_codes_dataset = datasets.load_from_disk(dataset_path)

NUM_VECTORS_IN_CODEBOOK = 1024
NUM_TRAIN_TIMESTEPS = 300

Q_transitioning = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])

for item in tqdm(noisy_codes_dataset):
    noisy_audio_codes = item['noisy_audio_codes']
    clean_audio_codes = item['clean_audio_codes']
    timestep = item['timesteps']

    Q_transitioning[timestep, noisy_audio_codes, clean_audio_codes] += 1


file_name = f'Q_transitioning_raw_{NUM_TRAIN_TIMESTEPS}_timesteps.pth'
torch.save(Q_transitioning, file_name)
print("file saved:", file_name)