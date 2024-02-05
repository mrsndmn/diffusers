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
from diffusers.schedulers.scheduling_vq_diffusion import VQDiffusionSchedulerDummyQPosterior
from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot
from transformers import DefaultDataCollator

audio_mnist_dataset_24khz_processed = datasets.load_from_disk("./audio_mnist_full_encodec_processed")

encodec_model_name = "facebook/encodec_24khz"

encodec_model = EncodecModel.from_pretrained(encodec_model_name).eval()
encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)

target_dir = "/tmp/validate_samples"

shutil.rmtree(target_dir, ignore_errors=True)
Path(target_dir).mkdir(exist_ok=True)

NUM_VECTORS_IN_CODEBOOK = 1024

with torch.no_grad():
    scheduler = VQDiffusionSchedulerDummyQPosterior(
        num_vec_classes=NUM_VECTORS_IN_CODEBOOK + 1,
        num_train_timesteps=100,
        device='cpu',
    )

    start_timestep = 25
    num_samples = 10

    # audio_mnist_dataset_24khz_processed_split = audio_mnist_dataset_24khz_processed.train_test_split(test_size=num_samples, seed=2)
    # audio_mnist_dataset_24khz_processed_test = audio_mnist_dataset_24khz_processed_split['test']

    collator = DefaultDataCollator()
    audio_mnist_dataset_24khz_processed_test = audio_mnist_dataset_24khz_processed # .select(range(0, len(audio_mnist_dataset_24khz_processed), 500))
    print("audio_mnist_dataset_24khz_processed_test", len(audio_mnist_dataset_24khz_processed_test))
    print("audio_mnist_dataset_24khz_processed_test label", audio_mnist_dataset_24khz_processed_test['label'])
    test_dataloader = torch.utils.data.DataLoader(
        audio_mnist_dataset_24khz_processed_test,
        collate_fn=collator,
        batch_size=num_samples,
        shuffle=True,
        # pin_memory=True,
        # num_workers=2,
    )

    test_batch = next(iter(test_dataloader))
    audio_codes_to_evaluate_start = test_batch['audio_codes']
    print("test_batch['labels']", test_batch['labels'])

    audio_codes_to_evaluate_start = audio_codes_to_evaluate_start.reshape([num_samples, -1])

    log_one_hot_x_0_probas = index_to_log_onehot(audio_codes_to_evaluate_start, scheduler.num_embed)

    timesteps = start_timestep * torch.ones(audio_codes_to_evaluate_start.shape[0], dtype=torch.long, device=audio_codes_to_evaluate_start.device)
    start_samples = scheduler.add_noise(
        log_one_hot_x_0_probas,
        timesteps,
    )['sample']

    print("changed_tokens", (audio_codes_to_evaluate_start != start_samples).sum())
    print("total_tokens", audio_codes_to_evaluate_start.numel())

    for i in range(num_samples):
        for token_type in ['noisy', 'clean']:
            if token_type == 'noisy':
                audio_codes = start_samples[i]
            else:
                audio_codes = audio_codes_to_evaluate_start[i]

            audio_codes = audio_codes.reshape(-1, 4)
            audio_codes = audio_codes.permute(1, 0)
            audio_codes = audio_codes.unsqueeze(0).unsqueeze(1)

            padding_mask = None
            print("audio_code", audio_codes.shape)
            print("audio_codes masked", (audio_codes == scheduler.mask_class).sum())
            audio_codes[audio_codes == scheduler.mask_class] = 0
            audio_wave = encodec_model.decode(audio_codes, [None], padding_mask, return_dict=True).audio_values
            print("audio_wave", audio_wave.shape)

            current_text_condition = str(test_batch['labels'][i].item())

            SAMPLE_RATE = 24000

            file_name = f"{target_dir}/{token_type}_{i}_{current_text_condition}.wav"
            torchaudio.save(file_name, audio_wave[0].to('cpu'), sample_rate=SAMPLE_RATE)
            print("saved", file_name)


