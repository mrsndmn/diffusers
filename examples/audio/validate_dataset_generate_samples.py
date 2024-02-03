import shutil
import torch
import datasets
import sys
import torchaudio
from pathlib import Path

if torch.backends.mps.is_available():
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/frechet-audio-distance')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/diffusers/src')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/transformers/src')
else:
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

from transformers import EncodecModel, AutoProcessor

audio_mnist_dataset_24khz_processed = datasets.load_from_disk("./audio_mnist_full_encodec_processed")

encodec_model_name = "facebook/encodec_24khz"

encodec_model = EncodecModel.from_pretrained(encodec_model_name).eval()
encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)

target_dir = "/tmp/validate_samples"

shutil.rmtree(target_dir, ignore_errors=True)
Path(target_dir).mkdir(exist_ok=True)

with torch.no_grad():
    for i in range(10):
        dataset_item = audio_mnist_dataset_24khz_processed[i]
        print("dataset_item", dataset_item)

        audio_codes = torch.tensor(dataset_item['audio_codes'])
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes.permute(1, 0)
        audio_codes = audio_codes.unsqueeze(0).unsqueeze(1)

        padding_mask = None
        print("audio_code", audio_codes.shape)
        audio_wave = encodec_model.decode(audio_codes, [None], padding_mask, return_dict=True).audio_values
        print("audio_wave", audio_wave.shape)

        current_text_condition = str(dataset_item['label'])

        SAMPLE_RATE = 24000

        file_name = f"{target_dir}/{i}_{current_text_condition}.wav"
        torchaudio.save(file_name, audio_wave[0].to('cpu'), sample_rate=SAMPLE_RATE)
        print("saved", file_name)


