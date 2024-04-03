import sys

import torch

print("torch.cuda.is_available()", torch.cuda.is_available())

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

import os
import torchaudio

from datetime import datetime

import datasets
from datasets import Audio, concatenate_datasets
from transformers import EncodecModel, AutoProcessor, DefaultDataCollator
from diffusers import UNet1DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMEncodecCodebookPipeline, DDPMAudioCodesPipeline, DDPMAudioCodesProbasPipeline

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os


MAX_AUDIO_CODES_LENGTH = 192
MAX_TRAIN_SAMPLES = 10
NUM_TRAIN_TIMESTEPS = 1000

@dataclass
class TrainingConfig:
    sample_size = 128  # the generated image resolution
    train_batch_size = 512
    eval_batch_size = 64  # how many images to sample during evaluation
    num_epochs = 10000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 100
    save_image_epochs = 200
    save_model_epochs = 200
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-audio-mnist-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def process_audio_encodec(encodec_processor, encodec_model, examples):

    audio_codes_batch = []
    audio_embeddings_batch = []

    for audio in examples['audio']:
        example = {
            "audio": {
                "array": audio['array']
            }
        }
        audio_processed = _process_audio_encodec(encodec_processor, encodec_model, example)
        audio_codes_batch.append( audio_processed['audio_codes'] )
        audio_embeddings_batch.append( audio_processed['audio_embeddings'] )

    return {
        "audio_codes": audio_codes_batch,
        "audio_embeddings": audio_embeddings_batch,
    }

def _process_audio_encodec(encodec_processor, encodec_model, example):

    # print("example['audio']['array']", example['audio']['array'])
    raw_audio = torch.tensor(example['audio']['array'])
    # print("raw_audio", raw_audio.shape)
    result = encodec_processor(raw_audio=raw_audio, sampling_rate=encodec_processor.sampling_rate, return_tensors="pt")
    encoder_outputs = encodec_model.encode(result["input_values"].to(encodec_model.device), result["padding_mask"].to(encodec_model.device))

    audio_codes = encoder_outputs.audio_codes
    audio_codes = audio_codes[0].repeat(1, 1, 10)
    audio_codes = audio_codes[0, :1, :MAX_AUDIO_CODES_LENGTH]
    audio_embeddings = encodec_model.quantizer.decode(audio_codes.unsqueeze(0))[0]

    # print("audio_codes processed", audio_codes.shape)
    return {
        "audio_codes": audio_codes,
        "audio_embeddings": audio_embeddings
    }

    precision = 1
    audio_codes = (encoder_outputs.audio_codes / precision).to(dtype=torch.long)
    codebook_size = encodec_model.config.codebook_size / precision

    audio_codes_norm = (audio_codes * 2 / codebook_size) - 1
    audio_codes_norm = audio_codes_norm[0].repeat(1, 1, 10)

    assert audio_codes_norm.shape[-1] > MAX_AUDIO_CODES_LENGTH

    # padding_length = MAX_AUDIO_CODES_LENGTH - encoder_outputs.audio_codes.shape[-1]
    # assert padding_length > 0, f"padding_length={padding_length}, encoder_outputs.audio_codes.shape[-1]={encoder_outputs.audio_codes.shape[-1]}"
    # print("padding_length", padding_length)

    return {
        # **result,
        # "padding_mask": result['padding_mask'][0],
        "audio_codes": audio_codes_norm[0, :1, :MAX_AUDIO_CODES_LENGTH],
        # "input_values": result["input_values"][0],
    }

@torch.no_grad()
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    audio_codes = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).audio_codes
    # print("audio_codes.shape", audio_codes.shape)
    # print("audio_codes", audio_codes)

    encodec_model_device = encodec_model.to(device=audio_codes.device)

    all_audio_values = []
    for audio_code_i in range(audio_codes.shape[0]):
        audio_code = audio_codes[audio_code_i].unsqueeze(0).unsqueeze(0)
        # print("audio_code.shape", audio_code.shape)
        # padding_mask = (audio_code != 0).to(device=audio_code.device)
        padding_mask = None

        # print("padding_mask", padding_mask)
        print_tensor_statistics("audio_code", audio_code)
        audio_values = encodec_model_device.decode(audio_code, [None], padding_mask, return_dict=True).audio_values
        # print(audio_code_i, "audio_code", audio_code.shape)
        # print(audio_code_i, "audio_values", audio_values.shape)
        all_audio_values.append(audio_values)

    all_audio_values = torch.cat(all_audio_values, dim=0)
    # print("all_audio_values.shape", all_audio_values.shape)

    # Make a grid out of the images
    # image_grid = make_grid(images, rows=4, cols=4)
    # print("audio_values", audio_values.shape)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    for i in range(all_audio_values.shape[0]):
        audio_wave = all_audio_values[i]
        # print("audio_wave", audio_wave.shape)
        torchaudio.save(f"{test_dir}/{epoch}_{i}.wav", audio_wave.to('cpu'), sample_rate=24000)

    print(f"evaluate for epoch {epoch} done")

    return {
        "eval_codes/min": all_audio_values.min(),
        "eval_codes/max": all_audio_values.max(),
        "eval_codes/median": all_audio_values.median(),
        "eval_codes/mean": all_audio_values.float().mean(),
    }
    # image_grid.save()

def print_tensor_statistics(tensor_name, tensor):
    print(f"{tensor_name} [{tensor.shape}]: min={tensor.min():.4f}, max={tensor.max():.4f}, median={tensor.median():.4f}, mean={tensor.float().mean():.4f}")


def train_loop(config, model, encodec_model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            raise Exception("push_to_hub option is not supported")
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example_" + str(datetime.now()))

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            audio_codes = batch["audio_codes"] # [ bs, num_channels, sequence_length ]
            audio_embeddings = batch["audio_embeddings"]
            print_tensor_statistics("audio_embeddings      ", audio_embeddings)
            print_tensor_statistics("audio_codes           ", audio_codes)

            # Sample noise to add to the images
            bs = audio_codes.shape[0]
            channels = audio_codes.shape[1]
            assert channels == 1, f'channels != 1: {audio_codes.shape}'
            seq_len = audio_codes.shape[2]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=audio_embeddings.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            # >>> torch.zeros([5, 1, 3]).scatter( 2, torch.tensor( [ [ [ 0 ] ], [ [ 1 ] ], [ [ 2 ] ] ] ), torch.ones(5,1,3) )
            # tensor([[[1., 0., 0.]],
            #         [[0., 1., 0.]],
            #         [[0., 0., 1.]],
            #         [[0., 0., 0.]],
            #         [[0., 0., 0.]]])


            noise = torch.randn(audio_embeddings.shape).to(audio_embeddings.device)

            noisy_audio_embeddings = noise_scheduler.add_noise(audio_embeddings, noise, timesteps)
            print_tensor_statistics("noisy_audio_embeddings", noisy_audio_embeddings)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_audio_embeddings, timesteps, return_dict=False)[0]

                print_tensor_statistics("noise_pred            ", noise_pred)
                print_tensor_statistics("noise                 ", noise)

                assert noise_pred.shape == noise.shape, f"{noise_pred.shape} == {noise.shape}"
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMEncodecCodebookPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, encodec_model=encodec_model)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                logs_scalars = evaluate(config, epoch, pipeline)
                accelerator.log(logs_scalars, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    raise Exception("push_to_hub is not supported")
                else:
                    pipeline.save_pretrained(config.output_dir)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    # WARN label в этом датасете сейчас - идентификатор спикера!
    audio_mnist_dataset = datasets.load_from_disk("./audio_mnist_full")

    audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=24000))


    # librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    encodec_model_name = "facebook/encodec_24khz"

    encodec_model = EncodecModel.from_pretrained(encodec_model_name)
    encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)


    config = TrainingConfig()

    def process_audio(example):
        return process_audio_encodec(encodec_processor, encodec_model, example)

    audio_mnist_dataset_24khz_processed = concatenate_datasets([audio_mnist_dataset_24khz.select(range(1))] * 1024)
    audio_mnist_dataset_24khz_processed.set_transform(process_audio)

    collator = DefaultDataCollator()

    print("audio_mnist_dataset_24khz_processed", len(audio_mnist_dataset_24khz_processed))
    train_dataloader = torch.utils.data.DataLoader(audio_mnist_dataset_24khz_processed, collate_fn=collator, batch_size=config.train_batch_size, shuffle=True)

    model = UNet1DModel(
        sample_size=128,  # the target image resolution
        in_channels=128,
        extra_in_channels=16,
        out_channels=128,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(256, 512, 1024),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock1DNoSkip",
            "DownBlock1D",
            "AttnDownBlock1D",
        ),
        mid_block_type='UNetMidBlock1D',
        up_block_types=(
            "AttnUpBlock1D",
            "UpBlock1D",
            "UpBlock1DNoSkip",
        ),
        out_block_type='OutConv1DBlock',
        act_fn='silu',
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_eval = model.to(device)
    model_eval.eval()
    encodec_model = encodec_model.to(device)
    encodec_model.eval()

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("model params:", count_params(model))

    train_loop(config, model, encodec_model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
