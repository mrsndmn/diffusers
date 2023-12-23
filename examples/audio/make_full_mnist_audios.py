from datasets import load_dataset, concatenate_datasets, Audio
import datasets
import torchaudio
import torch
from tqdm.auto import tqdm

SAMPLE_RATE = 24000

base_path = 'audio_mnist_full/audios/'

audio_mnist_dataset = datasets.load_from_disk("./audio_mnist_full")

audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

for i, sample in tqdm(enumerate(audio_mnist_dataset_24khz)):
    audio_t = torch.tensor(sample['audio']['array']).unsqueeze(0)
    print("audio_t.shape", audio_t.shape)
    torchaudio.save(base_path + str(i) + '.wav', audio_t, sample_rate=SAMPLE_RATE)
