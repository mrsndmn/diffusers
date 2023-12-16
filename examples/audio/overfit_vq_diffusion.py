import sys

import torch

from datetime import datetime
print("torch.cuda.is_available()", torch.cuda.is_available())
print("start", datetime.now())

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

import os
import torchaudio


import datasets
from datasets import Audio, concatenate_datasets
from transformers import EncodecModel, AutoProcessor, DefaultDataCollator, CLIPTextModel, AutoTokenizer
from diffusers import VQDiffusionScheduler, Transformer2DModel, VQDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers.pipelines.vq_diffusion.pipeline_vq_diffusion import VQDiffusionAudioUnconditionalPipeline

from dataclasses import dataclass
import torch

from accelerate import Accelerator
from tqdm.auto import tqdm
import os

from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot, multinomial_kl, log_categorical


MAX_AUDIO_CODES_LENGTH = 256
MAX_TRAIN_SAMPLES = 10
NUM_TRAIN_TIMESTEPS = 100
BANDWIDTH = 3.0
@dataclass
class TrainingConfig:
    sample_size = MAX_AUDIO_CODES_LENGTH  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 10000
    gradient_accumulation_steps = 1
    learning_rate = 3e-4
    lr_warmup_steps = 1000
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-audio-mnist-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    auxiliary_loss_weight = 0.01


def process_audio_encodec(encodec_processor, encodec_model, clip_tokenizer, examples):

    audio_codes_batch = []
    audio_scales_batch = []
    audio_embeddings_batch = []

    for audio in examples['audio']:
        example = {
            "audio": {
                "array": audio['array']
            }
        }
        audio_processed = _process_audio_encodec(encodec_processor, encodec_model, example)
        audio_codes_batch.append( audio_processed['audio_codes'] )
        audio_scales_batch.append( audio_processed['audio_scales'] )
        audio_embeddings_batch.append( audio_processed['audio_embeddings'] )

    # print(examples["label"])
    clip_processed = clip_tokenizer([ str(label) for label in examples["label"]], padding=True, return_tensors="pt")
    return {
        "audio_codes": audio_codes_batch,
        # "audio_scales": audio_scales_batch,
        "audio_embeddings": audio_embeddings_batch,
        **clip_processed,
    }

def _process_audio_encodec(encodec_processor, encodec_model: EncodecModel, example):

    # print("example['audio']['array']", example['audio']['array'])
    raw_audio = torch.tensor(example['audio']['array'])
    # print("raw_audio", raw_audio.shape)
    result = encodec_processor(raw_audio=raw_audio, sampling_rate=encodec_processor.sampling_rate, return_tensors="pt")
    encoder_outputs = encodec_model.encode(
        result["input_values"].to(encodec_model.device),
        result["padding_mask"].to(encodec_model.device),
        bandwidth=BANDWIDTH,
    )

    audio_codes = encoder_outputs.audio_codes
    # print("1 audio_codes.shape", audio_codes.shape)
    audio_codes = audio_codes[0].permute(0, 2, 1)
    audio_codes = audio_codes.reshape(audio_codes.shape[0], -1)
    audio_codes = audio_codes.unsqueeze(1)
    audio_codes = audio_codes.repeat(1, 1, 20)
    # print("2 audio_codes.shape", audio_codes.shape)
    audio_codes = audio_codes[0, :, :MAX_AUDIO_CODES_LENGTH]
    # print("3 audio_codes.shape", audio_codes.shape)
    audio_embeddings = encodec_model.quantizer.decode(audio_codes.unsqueeze(0))[0]

    # print("audio_codes processed", audio_codes.shape)
    # print("encoder_outputs.audio_scales", encoder_outputs.audio_scales)
    return {
        "audio_codes": audio_codes,
        "audio_scales": encoder_outputs.audio_scales,
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
def evaluate(config, epoch, pipeline: VQDiffusionAudioUnconditionalPipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    text_condition = [ str(x) for x in range(10) ]
    pipeline_out = pipeline(
        num_inference_steps=NUM_TRAIN_TIMESTEPS,
        bandwidth=BANDWIDTH,
        num_generated_audios = 10,
        text_condition=text_condition,
    )
    audio_codes = pipeline_out.audio_codes
    audio_values = pipeline_out.audio_values
    print("audio_codes.shape", audio_codes.shape)
    print("audio_values.shape", audio_values.shape)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    for i in range(audio_values.shape[0]):
        current_text_condition = text_condition[i]
        audio_wave = audio_values[i]
        torchaudio.save(f"{test_dir}/{epoch}_{current_text_condition}.wav", audio_wave.to('cpu'), sample_rate=24000)

    print(f"evaluate for epoch {epoch} done")

    return {
        "eval_codes/min":    audio_codes.min(),
        "eval_codes/max":    audio_codes.max(),
        "eval_codes/median": audio_codes.median(),
        "eval_codes/mean":   audio_codes.float().mean(),
    }
    # image_grid.save()

def print_tensor_statistics(tensor_name, tensor):
    print(f"{tensor_name} [{tensor.shape}] [{tensor.device}]: min={tensor.min():.4f}, max={tensor.max():.4f}, median={tensor.median():.4f}, mean={tensor.float().mean():.4f}")


def train_loop(
    config: TrainingConfig,
    model: Transformer2DModel,
    clip_tokenizer: AutoTokenizer,
    clip_text_model: CLIPTextModel,
    encodec_model: EncodecModel,
    noise_scheduler: VQDiffusionScheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    ):
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
    model.train()
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

            # Sample noise to add to the images
            bs = audio_codes.shape[0]
            channels = audio_codes.shape[1]
            assert channels == 1, f'channels != 1: {audio_codes.shape}'
            seq_len = audio_codes.shape[2]

            audio_codes = audio_codes.reshape([bs, -1])
            print_tensor_statistics("audio_embeddings      ", audio_embeddings)
            print_tensor_statistics("audio_codes           ", audio_codes)

            clip_outputs = clip_text_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

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

            log_one_hot_audio_codes = index_to_log_onehot(audio_codes, noise_scheduler.num_embed)
            print_tensor_statistics("log_one_hot_audio_codes", log_one_hot_audio_codes)

            noisy_audio_codes = noise_scheduler.add_noise(log_one_hot_audio_codes, timesteps)
            print_tensor_statistics("noisy_audio_codes", noisy_audio_codes)
            print_tensor_statistics("timesteps        ", timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual

                logs = {}

                log_x0_reconstructed = model(
                    hidden_states=noisy_audio_codes,
                    encoder_hidden_states=clip_outputs.last_hidden_state,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                log_zero_column = -70 * torch.ones([ log_x0_reconstructed.shape[0], 1, log_x0_reconstructed.shape[-1] ], device=log_x0_reconstructed.device)
                log_x0_reconstructed = torch.cat([log_x0_reconstructed, log_zero_column], dim=1)
                log_x0_reconstructed = torch.clamp(log_x0_reconstructed, -70, 0)

                print_tensor_statistics("log_x0_reconstructed ", log_x0_reconstructed)

                log_model_prob_x_t_min_1 = noise_scheduler.q_posterior_new(log_p_x_0=log_x0_reconstructed, x_t=noisy_audio_codes, t=timesteps)

                print_tensor_statistics("log_model_prob_x_t_min_1 ", log_model_prob_x_t_min_1)

                # log_p_x_0 = log_one_hot_audio_codes[:, :-1, :]
                log_true_prob_x_t_min_1 = noise_scheduler.q_posterior_new(log_p_x_0=log_one_hot_audio_codes, x_t=noisy_audio_codes, t=timesteps)
                print_tensor_statistics("log_true_prob_x_t_min_1       ", log_true_prob_x_t_min_1)

                kl_loss = multinomial_kl(log_true_prob_x_t_min_1, log_model_prob_x_t_min_1)
                print_tensor_statistics("kl_loss       ", kl_loss)

                kl_loss_sum_pixels = kl_loss.mean(dim=-1)
                print_tensor_statistics("kl_loss_sum_pixels ", kl_loss_sum_pixels)

                # L_{0}
                decoder_x0_nll = - log_categorical(log_one_hot_audio_codes, log_model_prob_x_t_min_1)
                decoder_x0_nll = decoder_x0_nll.mean(dim=-1)

                non_zero_timesteps = (timesteps != 0)
                decoder_x0_nll[non_zero_timesteps] = 0
                print_tensor_statistics("decoder_nll ", decoder_x0_nll)

                zero_timesteps = ~non_zero_timesteps
                kl_loss_sum_pixels[zero_timesteps] = 0
                print_tensor_statistics("kl_loss_sum_pixels zeroed zero t ", kl_loss_sum_pixels)
                result_loss = (kl_loss_sum_pixels + decoder_x0_nll).mean()

                if config.auxiliary_loss_weight > 0:
                    log_one_hot_audio_codes_no_mask = log_one_hot_audio_codes[:, :-1, :]
                    log_x0_reconstructed_no_mask = log_x0_reconstructed[:,:-1,:]
                    assert log_one_hot_audio_codes_no_mask.shape == log_x0_reconstructed_no_mask.shape, f"auxiliary loss shapes mismatch: {log_one_hot_audio_codes_no_mask.shape} != {log_x0_reconstructed_no_mask.shape}"
                    kl_aux = multinomial_kl(log_one_hot_audio_codes_no_mask, log_x0_reconstructed_no_mask)
                    kl_aux = kl_aux.mean(dim=-1)
                    print_tensor_statistics("kl_aux ", kl_aux)
                    kl_aux = kl_aux.mean() * config.auxiliary_loss_weight

                    result_loss += kl_aux

                accelerator.backward(result_loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            progress_bar.update(1)

            base_logs = {}
            base_logs["loss/loss"]        =  result_loss.detach().item()
            base_logs["loss/kl_loss"]     =  kl_loss_sum_pixels.mean().detach().item()
            base_logs["loss/decoder_nll"] =  decoder_x0_nll.mean().detach().item()
            if config.auxiliary_loss_weight > 0:
                base_logs["loss/kl_aux"] = kl_aux.detach().item()

            base_logs["params/lr"]        =  lr_scheduler.get_last_lr()[0]
            base_logs["params/step"]      =  global_step

            progress_bar.set_postfix(**base_logs)

            logs.update(base_logs)

            logs["system/gpu_memory_allocated"] = torch.cuda.memory_allocated()

            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = VQDiffusionAudioUnconditionalPipeline(
                    encodec=encodec_model,
                    clip_tokenizer=clip_tokenizer,
                    clip_text_model=clip_text_model,
                    transformer=unwrapped_model,
                    scheduler=noise_scheduler,
                )
                logs_scalars = evaluate(config, epoch, pipeline)
                accelerator.log(logs_scalars, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    raise Exception("push_to_hub is not supported")
                else:
                    unwrapped_model.save_pretrained(config.output_dir)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    print("loading dataset", datetime.now())
    audio_mnist_dataset = datasets.load_from_disk("./audio_mnist_full")

    audio_mnist_dataset_24khz = audio_mnist_dataset.cast_column("audio", Audio(sampling_rate=24000))
    print("loaded and casted dataset", datetime.now())


    # librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    encodec_model_name = "facebook/encodec_24khz"

    encodec_model = EncodecModel.from_pretrained(encodec_model_name)
    encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
    print("encodec loaded", datetime.now())

    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    config = TrainingConfig()

    def process_audio(example):
        return process_audio_encodec(encodec_processor, encodec_model, clip_tokenizer, example)

    NUM_VECTORS_IN_CODEBOOK = 1024

    print("creating processed dataset", datetime.now())
    # audio_mnist_dataset_24khz_processed = concatenate_datasets([audio_mnist_dataset_24khz.select(range(10))] * NUM_VECTORS_IN_CODEBOOK * 10)
    audio_mnist_dataset_24khz_processed = audio_mnist_dataset_24khz.select(range(1024))
    audio_mnist_dataset_24khz_processed.set_transform(process_audio)
    print("created processed dataset", datetime.now())

    collator = DefaultDataCollator()

    print("audio_mnist_dataset_24khz_processed", len(audio_mnist_dataset_24khz_processed))
    train_dataloader = torch.utils.data.DataLoader(audio_mnist_dataset_24khz_processed, collate_fn=collator, batch_size=config.train_batch_size, shuffle=True)

    height = 1
    width = MAX_AUDIO_CODES_LENGTH
    model_kwargs = {
        "attention_bias": True,
        "cross_attention_dim": clip_text_model.config.hidden_size,
        "attention_head_dim": height * width,
        "num_attention_heads": 8,
        "num_vector_embeds": NUM_VECTORS_IN_CODEBOOK + 1,
        "num_embeds_ada_norm": NUM_TRAIN_TIMESTEPS,
        "norm_num_groups": 32,
        "sample_size": width,
        "num_layers": 6,
        "activation_fn": "geglu-approximate",
    }

    model = Transformer2DModel(**model_kwargs)
    assert model.is_input_continuous == False, 'transformer is discrete'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    noise_scheduler = VQDiffusionScheduler(
        num_vec_classes=model_kwargs['num_vector_embeds'],
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )


    encodec_model = encodec_model.to(device)
    encodec_model.eval()

    clip_text_model = clip_text_model.to(device)
    clip_text_model.eval()

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("model params:", count_params(model))

    train_loop(config, model, clip_tokenizer, clip_text_model, encodec_model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
