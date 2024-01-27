import argparse
import sys

import torch
import torch.nn.functional as F

from datetime import datetime
print("torch.cuda.is_available()", torch.cuda.is_available())
print("start", datetime.now())

import time
import yaml

from pathlib import Path

# todo remove hardcode
if torch.backends.mps.is_available():
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/frechet-audio-distance')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/diffusers/src')
    sys.path.insert(0, '/Users/d.tarasov/workspace/hse/transformers/src')
else:
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
    sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

from frechet_audio_distance import FrechetAudioDistance
import os
import torchaudio

import datasets
from transformers import EncodecModel, AutoProcessor, DefaultDataCollator, CLIPTextModel, AutoTokenizer
from diffusers import Transformer2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers.pipelines.deprecated.vq_diffusion.pipeline_vq_diffusion import VQDiffusionAudioTextConditionalPipeline

from dataclasses import dataclass
import torch

from accelerate import Accelerator
from tqdm.auto import tqdm
import os

from timestep_sampling import TimestepsSampler

from audio_mnist_classifier import AudioMNISTModel, get_spectrogram

from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot, multinomial_kl, log_categorical, VQDiffusionDenseTrainedDummyQPosteriorScheduler, VQDiffusionSchedulerDummyQPosterior, VQDiffusionScheduler

NUM_VECTORS_IN_CODEBOOK = 1024
MAX_AUDIO_CODES_LENGTH = 256
MAX_TRAIN_SAMPLES = 10
SAMPLE_RATE = 24000
BANDWIDTH = 3.0

script_start_time = datetime.now()

@dataclass
class TrainingConfig:

    sample_size = MAX_AUDIO_CODES_LENGTH  # the generated image resolution

    num_train_timesteps = 100

    # dataset and iteration
    dataset_path = "./audio_mnist_full_encodec_processed"
    train_batch_size = 100
    # eval_batch_size = 20  # how many images to sample during evaluation
    num_epochs = 10000
    transformer_dropout = 0.0

    # optimizer
    learning_rate = 1e-4
    lr_warmup_steps = 5000
    gradient_accumulation_steps = 1

    # save strategy
    save_image_epochs = 14
    save_model_epochs = 14

    num_evaluation_samples = 100

    # accelerator configs
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0

    experiment_name = "anon"
    output_dir = "ddpm-audio-mnist-128"  # the model name locally and on the HF Hub

    # losses
    auxiliary_loss_weight = 0.01
    kl_loss_weight = 0.1
    decoder_nll_loss_weight = 1.0

    timesteps_sampling = TimestepsSampler.SAMPLING_STRATEGY_UNIFORM

    noise_scheduler = "dense_trained_dummy_q_posterior" # optimized_masked_unifor | optimized_masked_uniform_dummy_q_posterior | dense_trained_dummy_q_posterior
    # used for dense_trained
    noise_scheduler_q_transition_martices_path = "./Q_transitioning_normed.pth"
    noise_scheduler_q_transition_cummulative_martices_path = "./Q_transitioning_cumulative_norm.pth"

    noise_scheduler_q_transition_transposed_martices_path = "./Q_transitioning_transposed_normed.pth"
    noise_scheduler_q_transition_transposed_cummulative_martices_path = "./Q_transitioning_cumulative_transposed_normed.pth"

@torch.no_grad()
def evaluate(config: TrainingConfig, epoch, pipeline: VQDiffusionAudioTextConditionalPipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    evaluate_pipeline_counter = time.perf_counter()
    condition_classes = list(range(10)) * 10
    text_condition = [ str(x) for x in condition_classes ]
    pipeline_out = pipeline(
        num_inference_steps=config.num_train_timesteps,
        bandwidth=BANDWIDTH,
        num_generated_audios = len(condition_classes),
        text_condition=text_condition,
    )
    audio_codes = pipeline_out.audio_codes
    audio_values = pipeline_out.audio_values
    print("audio_codes.shape", audio_codes.shape)
    print("audio_values.shape", audio_values.shape) # [bs, n_channels, waveform_length]
    timings_evaluate_pipeline = time.perf_counter() - evaluate_pipeline_counter

    # Save the images
    attentions_norms_by_timesteps = {}
    for timestep_i in list(range(0, config.num_train_timesteps, 20)) + [ config.num_train_timesteps - 3, config.num_train_timesteps - 2, config.num_train_timesteps - 1]:
        total_cross_attention_norm = 0
        total_self_attention_norm = 0
        for transformer_block_j in range(len(pipeline_out.self_attentions[0])):
            self_attention = pipeline_out.self_attentions[timestep_i][transformer_block_j]
            cross_attention = pipeline_out.cross_attentions[timestep_i][transformer_block_j]
            #
            total_cross_attention_norm += self_attention.norm(2)
            total_self_attention_norm  += cross_attention.norm(2)
        #
        attentions_norms_by_timesteps[f'eval_metrics_attention/self_norm_minus_cross_norm_{timestep_i}'] = (total_self_attention_norm - total_cross_attention_norm).item()
        attentions_norms_by_timesteps[f'eval_metrics_attention/sum_cross_attention_norm_{timestep_i}'] = total_cross_attention_norm.item()
        attentions_norms_by_timesteps[f'eval_metrics_attention/sum_self_attention_norm_{timestep_i}'] = total_self_attention_norm.item()

    print(f"{timestep_i}\t self - cross", total_self_attention_norm - total_cross_attention_norm)

    evaluate_save_samples_counter = time.perf_counter()
    test_dir = os.path.join(config.output_dir, config.experiment_name, "samples")
    generated_samples_path = f"{test_dir}/{script_start_time}/{epoch}"
    generated_samples_path = Path(generated_samples_path)
    generated_samples_path.mkdir(parents=True, exist_ok=True)

    for i in range(audio_values.shape[0]):
        current_text_condition = text_condition[i]
        audio_wave = audio_values[i]
        torchaudio.save(f"{generated_samples_path}/{current_text_condition}.wav", audio_wave.to('cpu'), sample_rate=SAMPLE_RATE)
    timings_evaluate_save_samples_counter = time.perf_counter() - evaluate_save_samples_counter

    evaluate_fad_counter = time.perf_counter()
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    fad_score = frechet.score(
        "audio_mnist_full/audios/",
        generated_samples_path,
        background_embds_path="audio_mnist_full/audios/frechet_embeddings.npy",
    )
    timings_evaluate_fad_counter = time.perf_counter() - evaluate_fad_counter
    print("fad_score", fad_score)

    evaluate_classifier_counter = time.perf_counter()
    audio_mnist_state_dict = torch.load("audio_mnist_classifier.pth", map_location='cpu')
    audio_mnist_classifier = AudioMNISTModel()
    audio_mnist_classifier.load_state_dict(audio_mnist_state_dict)
    audio_mnist_classifier.to(audio_values.device)
    audio_mnist_classifier.eval()

    audio_values_spectrograms = []
    new_sample_rate = 8000
    resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=new_sample_rate).to(audio_values.device)

    for i in range(audio_values.shape[0]):
        current_text_condition = text_condition[i]
        current_waveform = audio_values[i]
        resmpled_current_waveform = resampler(current_waveform)

        # torchaudio.save(f"{test_dir}/{script_start_timestep}/{epoch}/{current_text_condition}_for_classifier.wav", resmpled_current_waveform.to('cpu'), sample_rate=new_sample_rate)

        resmpled_current_waveform_np = resmpled_current_waveform[0].cpu().numpy()
        spectrogram_t = torch.tensor(get_spectrogram(resmpled_current_waveform_np, sample_rate=new_sample_rate), device=audio_values.device)
        audio_values_spectrograms.append(spectrogram_t.float())


    audio_values_spectrograms = torch.cat(audio_values_spectrograms, dim=0)
    audio_values_spectrograms = audio_values_spectrograms.unsqueeze(1)
    print("audio_values_spectrograms.shape", audio_values_spectrograms.shape)

    classiier_logits = audio_mnist_classifier(audio_values_spectrograms)
    classiier_probas = F.softmax(classiier_logits, dim=-1)
    classiier_classes = classiier_probas.max(dim=-1).indices
    print("classiier_classes", classiier_classes)

    negative_log_likelihood = F.cross_entropy(classiier_logits, torch.tensor(condition_classes, device=classiier_logits.device))
    eval_classifier_perplexity = torch.exp(negative_log_likelihood.detach()).item()
    print("eval_classifier_perplexity", eval_classifier_perplexity)

    eval_classifier_accuracy = (classiier_classes == torch.tensor(condition_classes, device=classiier_classes.device)).float().mean().item()
    timings_evaluate_classifier_counter = time.perf_counter() - evaluate_classifier_counter


    # todo make evaluation of audio mnist classifier with labels

    print(f"evaluate for epoch {epoch} done")

    return {
        "eval_codes/min":    audio_codes.min(),
        "eval_codes/max":    audio_codes.max(),
        "eval_codes/median": audio_codes.median(),
        "eval_codes/mean":   audio_codes.float().mean(),
        "eval_metrics/classifier_accuracy": eval_classifier_accuracy,
        "eval_metrics/classifier_perplexity": eval_classifier_perplexity,
        "eval_metrics/fad_score": fad_score,

        **attentions_norms_by_timesteps,

        "timings/evaluate_pipeline": timings_evaluate_pipeline,
        "timings/evaluate_save_samples": timings_evaluate_save_samples_counter,
        "timings/evaluate_fad": timings_evaluate_fad_counter,
        "timings/evaluate_classifier": timings_evaluate_classifier_counter,
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
    timesteps_sampler: TimestepsSampler,
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
        # device_placement=False,
    )

    if accelerator.is_main_process:
        if config.push_to_hub:
            raise Exception("push_to_hub option is not supported")
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        tracker_name = config.experiment_name + "_" + str(script_start_time)
        accelerator.init_trackers(tracker_name)

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

        masked_tokens_counts = []
        model.train()

        for step, batch in enumerate(train_dataloader):
            logs = {}

            train_iteration_counter = time.perf_counter()

            audio_codes = batch["audio_codes"] # [ bs, num_channels, sequence_length ]

            # Sample noise to add to the images
            bs = audio_codes.shape[0]
            channels = audio_codes.shape[1]
            assert channels == 1, f'channels != 1: {audio_codes.shape}'
            seq_len = audio_codes.shape[2]

            audio_codes = audio_codes.reshape([bs, -1])
            print_tensor_statistics("audio_codes           ", audio_codes)

            calc_clip_text_embeddings_counter = time.perf_counter()
            with torch.no_grad():
                print_tensor_statistics("input_ids", batch['input_ids'])
                print_tensor_statistics("attention_mask", batch['attention_mask'])

                clip_outputs = clip_text_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logs['timings/calc_clip_text_embeddings'] = time.perf_counter() - calc_clip_text_embeddings_counter

            # Sample a random timestep for each image
            timesteps = timesteps_sampler.sample(bs)

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

            add_noise_counter = time.perf_counter()
            scheduler_noise_output = noise_scheduler.add_noise(log_one_hot_audio_codes, timesteps)
            noisy_audio_codes = scheduler_noise_output['sample']

            logs['timings/add_noise'] = time.perf_counter() - add_noise_counter

            print_tensor_statistics("noisy_audio_codes", noisy_audio_codes)
            print_tensor_statistics("timesteps        ", timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual

                model_inference_counter = time.perf_counter()
                log_x0_reconstructed = model(
                    hidden_states=noisy_audio_codes,
                    encoder_hidden_states=clip_outputs.last_hidden_state,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                logs['timings/model_inference'] = time.perf_counter() - model_inference_counter
                print_tensor_statistics("log_x0_reconstructed model out", log_x0_reconstructed)
                print_tensor_statistics("timesteps", timesteps)
                print_tensor_statistics("clip_outputs.last_hidden_state", clip_outputs.last_hidden_state)

                if noise_scheduler.is_masked:
                    log_zero_column = -70 * torch.ones([ log_x0_reconstructed.shape[0], 1, log_x0_reconstructed.shape[-1] ], device=log_x0_reconstructed.device)
                    log_x0_reconstructed = torch.cat([log_x0_reconstructed, log_zero_column], dim=1)

                log_x0_reconstructed = torch.clamp(log_x0_reconstructed, -70, 0)

                calc_loss_counter = time.perf_counter()

                # if config.decoder_nll_loss_weight > 0.0 and config.kl_loss_weight > 0.0:
                if True:

                    print_tensor_statistics("log_x0_reconstructed ", log_x0_reconstructed)

                    # save do not save noise here
                    q_posterior_approximate_args = {
                        "log_p_x_0": log_x0_reconstructed,
                        "x_t":       noisy_audio_codes,
                        "t":         timesteps,
                    }

                    if noise_scheduler.is_masked:
                        log_model_prob_x_t_min_1 = noise_scheduler.q_posterior_orig(**q_posterior_approximate_args)
                    else:
                        log_model_prob_x_t_min_1 = noise_scheduler.q_posterior(**q_posterior_approximate_args)

                    print_tensor_statistics("log_model_prob_x_t_min_1 ", log_model_prob_x_t_min_1)

                    # log_p_x_0 = log_one_hot_audio_codes[:, :-1, :]
                    # but save noise here!

                    q_posterior_true_args = {
                        "log_p_x_0": log_one_hot_audio_codes,
                        "x_t":       noisy_audio_codes,
                        "t":         timesteps,
                    }

                    if noise_scheduler.is_masked:
                        log_true_prob_x_t_min_1 = noise_scheduler.q_posterior_orig(**q_posterior_true_args)
                    else:
                        log_true_prob_x_t_min_1 = noise_scheduler.q_posterior(**q_posterior_true_args)


                    print_tensor_statistics("log_true_prob_x_t_min_1       ", log_true_prob_x_t_min_1)

                    kl_loss = multinomial_kl(log_true_prob_x_t_min_1, log_model_prob_x_t_min_1)
                    print_tensor_statistics("kl_loss       ", kl_loss)

                    kl_loss = kl_loss.mean(dim=-1) # [ bs ]
                    timesteps_sampler.step(kl_loss, timesteps)
                    # возможно, это так importance sampling влияет на результат?

                    # kl_loss = kl_loss / timesteps_weight

                    non_zero_timesteps = (timesteps != 0)
                    zero_timesteps = ~non_zero_timesteps
                    kl_loss[zero_timesteps] = 0
                    print_tensor_statistics("kl_loss zeroed zero t ", kl_loss)

                    # sum due to each loss item has already been weighted with timesteps_weight
                    kl_loss = kl_loss.mean() * config.kl_loss_weight


                    # L_{0}
                    decoder_x0_nll = -log_categorical(log_one_hot_audio_codes, log_true_prob_x_t_min_1)

                    decoder_x0_nll = decoder_x0_nll.mean(dim=-1)

                    decoder_x0_nll[non_zero_timesteps] = 0
                    decoder_x0_nll = decoder_x0_nll * config.decoder_nll_loss_weight
                    print_tensor_statistics("decoder_nll ", decoder_x0_nll)

                    masked_tokens_count = (noisy_audio_codes == noise_scheduler.mask_class).sum().detach().cpu().item()
                    print("masked_tokens_count", masked_tokens_count)
                    masked_tokens_counts.append(masked_tokens_count)

                    result_loss = None # kl_loss + decoder_x0_nll.mean()
                else:
                    result_loss = None
                    kl_loss = None
                    decoder_x0_nll = None

                # kl_aux
                if noise_scheduler.is_masked:
                    log_one_hot_audio_codes_for_kl = log_one_hot_audio_codes[:, :-1, :]
                    log_x0_reconstructed_for_kl = log_x0_reconstructed[:,:-1,:]
                else:
                    log_one_hot_audio_codes_for_kl = log_one_hot_audio_codes
                    log_x0_reconstructed_for_kl = log_x0_reconstructed

                assert log_one_hot_audio_codes_for_kl.shape == log_x0_reconstructed_for_kl.shape, f"auxiliary loss shapes mismatch: {log_one_hot_audio_codes_for_kl.shape} != {log_x0_reconstructed_for_kl.shape}"
                kl_aux = multinomial_kl(log_one_hot_audio_codes_for_kl, log_x0_reconstructed_for_kl)
                kl_aux = kl_aux.mean(dim=-1) # [  bs ]
                # timesteps_sampler.step(kl_aux, timesteps)

                print_tensor_statistics("kl_aux ", kl_aux)
                kl_aux = kl_aux.mean() * config.auxiliary_loss_weight

                if result_loss is None:
                    result_loss = kl_aux
                else:
                    result_loss += kl_aux

                if result_loss.isnan().any():
                    raise ValueError("result_loss contains nan")

                logs['timings/calc_loss'] = time.perf_counter() - calc_loss_counter

                backward_counter = time.perf_counter()
                accelerator.backward(result_loss)
                logs['timings/backward'] = time.perf_counter() - backward_counter

                gradient_clipping_counter = time.perf_counter()
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                logs['timings/gradient_clipping'] = time.perf_counter() - gradient_clipping_counter

                optimizer_step_counter = time.perf_counter()
                optimizer.step()
                logs['timings/optimizer_step'] = time.perf_counter() - optimizer_step_counter

                lr_scheduler.step()
                optimizer.zero_grad()


            progress_bar.update(1)

            base_logs = {}
            base_logs["loss/loss"]        =  result_loss.detach().item()
            if kl_loss is not None and decoder_x0_nll is not None:
                base_logs["loss/kl_loss"]     =  kl_loss.detach().item()
                base_logs["loss/decoder_nll"] =  decoder_x0_nll.mean().detach().item()

            base_logs["loss/kl_aux"]      = kl_aux.detach().item()

            base_logs["params/lr"]    =  lr_scheduler.get_last_lr()[0]
            base_logs["params/step"]  =  global_step
            base_logs["params/epoch"] =  epoch

            progress_bar.set_postfix(**base_logs)

            logs.update(base_logs)

            logs['metrics/mean_masked_tokens_count'] = torch.tensor(masked_tokens_counts).float().mean().detach().cpu().item()

            if global_step % 100 == 0:
                logs["system/gpu_memory_allocated"] = torch.cuda.memory_allocated()

            logs['timings/train_iteration'] = time.perf_counter() - train_iteration_counter

            accelerator.log(logs, step=global_step)
            global_step += 1

            # go to start of iteration over train dataloader

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                with torch.no_grad():
                    pipeline = VQDiffusionAudioTextConditionalPipeline(
                        encodec=encodec_model,
                        clip_tokenizer=clip_tokenizer,
                        clip_text_model=clip_text_model,
                        transformer=unwrapped_model,
                        scheduler=noise_scheduler,
                    )

                    evaluate_start_counter = time.perf_counter()
                    logs_scalars = evaluate(config, epoch, pipeline)
                    logs_scalars['timings/evaluate_iteration'] = time.perf_counter() - evaluate_start_counter
                    accelerator.log(logs_scalars, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    raise Exception("push_to_hub is not supported")
                else:
                    variant = config.experiment_name + str(script_start_time)
                    unwrapped_model.save_pretrained(config.output_dir, variant=variant)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    # https://github.com/pytorch/pytorch/issues/40403#issuecomment-648439409
    # torch.multiprocessing.set_start_method('spawn')

    config = TrainingConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    print("args.config", args.config)

    with open(args.config, 'r') as file:
        print("override config params from", args.config)
        training_config_overwrite: dict = yaml.safe_load(file)
        for k, v in training_config_overwrite.items():
            print("override config param:", k, v)
            config.__setattr__(k, v)

    print("config.noise_scheduler", config.noise_scheduler)

    print("loading dataset", datetime.now())
    audio_mnist_dataset_24khz_processed = datasets.load_from_disk(config.dataset_path)
    # audio_mnist_dataset_24khz_processed = audio_mnist_dataset_24khz_processed.select(range(8))

    print("loaded and casted dataset", datetime.now())

    encodec_model_name = "facebook/encodec_24khz"

    encodec_model = EncodecModel.from_pretrained(encodec_model_name)
    encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
    print("encodec loaded", datetime.now())

    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    collator = DefaultDataCollator()

    print("audio_mnist_dataset_24khz_processed", len(audio_mnist_dataset_24khz_processed))
    train_dataloader = torch.utils.data.DataLoader(
        audio_mnist_dataset_24khz_processed,
        collate_fn=collator,
        batch_size=config.train_batch_size,
        shuffle=True,
        # pin_memory=True,
        # num_workers=2,
    )

    height = 1
    width = MAX_AUDIO_CODES_LENGTH
    model_kwargs = {
        "attention_bias": True,
        "cross_attention_dim": clip_text_model.config.hidden_size,
        "attention_head_dim": 96,
        "num_attention_heads": 6,
        "num_vector_embeds": NUM_VECTORS_IN_CODEBOOK+1,
        "num_embeds_ada_norm": config.num_train_timesteps,
        "sample_size": width,
        "height": height,
        "num_layers": 2,
        "activation_fn": "geglu-approximate",
        "output_attentions": True,
        "dropout": config.transformer_dropout,
    }

    print("model_kwargs", model_kwargs)

    model = Transformer2DModel(**model_kwargs)
    assert model.is_input_continuous == False, 'transformer is discrete'

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    # device = 'cpu'

    if config.noise_scheduler == "optimized_masked_uniform":
        noise_scheduler = VQDiffusionScheduler(
            num_vec_classes=model_kwargs['num_vector_embeds'],
            num_train_timesteps=config.num_train_timesteps,
            device=device,
        )
    elif config.noise_scheduler == "optimized_masked_uniform_dummy_q_posterior":
        noise_scheduler = VQDiffusionSchedulerDummyQPosterior(
            num_vec_classes=model_kwargs['num_vector_embeds'],
            num_train_timesteps=config.num_train_timesteps,
            device=device,
        )
    elif config.noise_scheduler == "dense_trained_dummy_q_posterior":
        noise_scheduler = VQDiffusionDenseTrainedDummyQPosteriorScheduler(
            q_transition_martices_path=config.noise_scheduler_q_transition_martices_path,
            q_transition_cummulative_martices_path=config.noise_scheduler_q_transition_cummulative_martices_path,
            q_transition_transposed_martices_path=config.noise_scheduler_q_transition_transposed_martices_path,
            q_transition_transposed_cummulative_martices_path=config.noise_scheduler_q_transition_transposed_cummulative_martices_path,
            device=device,
        )
    else:
        raise ValueError(f"unknown nooise scheduler: {config.noise_scheduler}")

    timesteps_sampler = TimestepsSampler(strategy=config.timesteps_sampling, num_timesteps=config.num_train_timesteps)
    timesteps_sampler.to(device)

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

    with torch.autograd.set_detect_anomaly(True, check_nan=True):
        train_loop(config, model, clip_tokenizer, clip_text_model, encodec_model, noise_scheduler, timesteps_sampler, optimizer, train_dataloader, lr_scheduler)
