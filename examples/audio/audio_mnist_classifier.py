import os
import librosa
import numpy as np

from tqdm import tqdm

import sys

import torch
import torch.nn as nn

if torch.backends.mps.is_available():
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/frechet-audio-distance')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/diffusers/src')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/transformers/src')
else:
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

import datasets
from datasets import Audio
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from collections import Counter

MAX_FRAMES = 192
N_FFT = 128

def get_spectrogram(waveform, sample_rate=8000):
    assert sample_rate==8000, "anothoer sample rate is not supported"

    assert len(waveform.shape) == 1, 'only single-channel waveforms are supported yet'

    converted = librosa.stft(waveform, n_fft=N_FFT)
    spectrum, _ = librosa.magphase(converted)
    spectrum = np.abs(spectrum).astype(np.float32)
    norm = spectrum.max()
    spectrum /= norm
    # print("spectrum", spectrum.shape)

    result = np.zeros((spectrum.shape[0], MAX_FRAMES))
    result[:spectrum.shape[0],:spectrum.shape[1]] = spectrum[:, :min(MAX_FRAMES, spectrum.shape[-1])]
    result = np.expand_dims(result, axis=0)

    return result # [ 1, MAX_FRAMES ]



def add_spectrogram_to_dataset(in_data):
    spectrogram = []
    # print("in_data", in_data)
    for elem in in_data['audio']:
        # print("elem", elem)
        elem = elem['array']

        spectrogram.append(get_spectrogram(elem))
    spectrogram = np.array(spectrogram, dtype=float)

    in_data["spectrogram"] = spectrogram
    return in_data


ENABLED_DEBUG = False

class DBG(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        if ENABLED_DEBUG:
            print(self.name, x.shape)
        return x

class AudioMNISTModel(nn.Module):

    def __init__(self):
        super().__init__()

        num_channels1 = 32
        num_channels2 = 64
        num_channels3 = 16

        self.model = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(1, num_channels1, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # [ bs, 32, 32, 95 ]
            DBG("maxpool2d 1"),
            # nn.BatchNorm2d(num_channels1),
            nn.Conv2d(num_channels1, num_channels2, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # [ bs, 16, 16, 46 ]
            DBG("maxpool2d 2"),
            nn.Conv2d(num_channels2, num_channels3, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # [ bs, 16, 16, 46 ]
            DBG("maxpool2d 3"),
            nn.Dropout(0.25),
            nn.Flatten(),
            DBG("flatten"),
            nn.Linear(2112, 128),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            # nn.Softmax(dim=-1),
            DBG("final"),
        )

        return

    def forward(self, spectrograms): # [ bs, 1, 65, 192 ]
        assert len(spectrograms.shape) == 4, "spectrograms shape dims is ok"
        assert spectrograms.shape[1] == 1, 'only one in channels is supported'

        return self.model(spectrograms)

def train_loop(
    model: AudioMNISTModel,
    optimizer: Adam,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs=10
    ):

    # train epoch
    loss_module = nn.CrossEntropyLoss()

    model.train()

    validate_accuracy = 0

    global_step = 0
    for epoch_num in range(num_epochs):

        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch_num}")

        for batch in train_dataloader:
            print("batch['spectrogram']", batch['spectrogram'].shape, "min", batch['spectrogram'].min(), "max", batch['spectrogram'].max())
            model_output = model(batch['spectrogram'])
            print("model_output", model_output.shape, "min", model_output.min(), "max", model_output.max())
            print("batch['label']", batch['label'].shape, "min", batch['label'].min(), "max", batch['label'].max())
            loss = loss_module(model_output, batch['label'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "step": global_step,
                "validate_accuracy": validate_accuracy,
            }
            progress_bar.set_postfix(**logs)

        with torch.no_grad():

            predicted_labels = []
            target_labels = []

            for batch in validation_dataloader:
                model.eval()
                model_output = model(batch['spectrogram'])
                loss = loss_module(model_output, batch['label'])

                predicted_probas = F.softmax(model_output, dim=-1)
                predicted_labels_batch = predicted_probas.max(dim=-1).indices

                predicted_labels.append(predicted_labels_batch)
                target_labels.append(batch['label'])

            predicted_labels = torch.cat(predicted_labels, dim=0)
            target_labels = torch.cat(target_labels, dim=0)

            assert predicted_labels.shape == target_labels.shape, 'labels shape is equal'

            validate_accuracy = (predicted_labels == target_labels).float().mean()
            print("validate_accuracy", validate_accuracy.item())



    return

if __name__ == '__main__':

    print("torch.__path__", torch.__path__)
    print("torch.backends.cudnn.version()", torch.backends.cudnn.version())

    audio_mnist_dataset = datasets.load_from_disk("./audio_mnist_full")
    audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=8000))
    audio_mnist_dataset_24khz = audio_mnist_dataset_24khz.map(lambda x: { 'label': int(os.path.basename(x['audio']['path']).split('_', maxsplit=1)[0]) })

    audio_mnist_dataset_sample = audio_mnist_dataset_24khz

    audio_mnist_labels_counts = Counter([x['label'] for x in audio_mnist_dataset_sample])
    print(audio_mnist_labels_counts)

    print(audio_mnist_dataset_24khz[0]['audio']['array'].shape)

    audio_mnist_dataset_sample = audio_mnist_dataset_sample.shuffle(0)

    audio_mnist_dataset_sample.set_transform(add_spectrogram_to_dataset)

    print("audio_mnist_dataset_sample[0]['spectrogram']", audio_mnist_dataset_sample[0]['spectrogram'].shape) # [ 1, n_spectrogram_features, num_frames ] ~ [ bs, 65, 16 ]
    print("audio_mnist_dataset_sample", audio_mnist_dataset_sample[0]['label'])

    # audio_mnist_dataset_sample # process padding

    # https://github.com/ARBasharat/AudioClassification/blob/master/AudioClassification2D.ipynb

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class TrainingConfig:
        train_batch_size = 64
        learning_rate = 1e-3

    config = TrainingConfig()

    model = AudioMNISTModel().to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    spectrogram = torch.tensor(audio_mnist_dataset_sample[0]['spectrogram'])
    spectrogram = spectrogram.unsqueeze(0).float()
    print("spectrogram", spectrogram.shape, spectrogram.dtype)

    result = model.forward(spectrogram.to(device))

    train_test_datasets = audio_mnist_dataset_sample.train_test_split(test_size=0.2)

    train_dataset = train_test_datasets['train']
    test_dataset  = train_test_datasets['test']

    def collate_fn(samples):

        labels = []
        spectrograms = []
        for sample in samples:
            labels.append(sample['label'])
            spectrograms.append(torch.tensor(sample['spectrogram']).unsqueeze(0))

        return {
            "label": torch.tensor( labels ).to(device),
            "spectrogram": torch.cat(spectrograms, dim=0).to(dtype=torch.float32).to(device)
        }

    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    validation_dataloader = DataLoader(test_dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    train_loop(model, optimizer, train_dataloader, validation_dataloader, num_epochs=10)

    torch.save(model.state_dict(), "audio_mnist_classifier.pth")


