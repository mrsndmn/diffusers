import torch

from tqdm import tqdm

from transformers import EncodecModel, AutoProcessor, DefaultDataCollator, CLIPTextModel, AutoTokenizer
from diffusers import VQDiffusionScheduler, Transformer2DModel, VQDiffusionPipeline
from diffusers.pipelines.vq_diffusion.pipeline_vq_diffusion import VQDiffusionAudioTextConditionalPipeline
from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot, multinomial_kl


encodec_model_name = "facebook/encodec_24khz"

encodec_model = EncodecModel.from_pretrained(encodec_model_name)
encodec_processor = AutoProcessor.from_pretrained(encodec_model_name)
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

NUM_VECTORS_IN_CODEBOOK = 7
MAX_AUDIO_CODES_LENGTH = 3
MAX_TRAIN_SAMPLES = 10
NUM_TRAIN_TIMESTEPS = 100
BANDWIDTH = 3.0
SAMPLE_RATE = 24000

height = 1
width = MAX_AUDIO_CODES_LENGTH

noise_scheduler = VQDiffusionScheduler(
    num_vec_classes=NUM_VECTORS_IN_CODEBOOK + 1,
    num_train_timesteps=NUM_TRAIN_TIMESTEPS,
)

pipeline = VQDiffusionAudioTextConditionalPipeline(
    encodec=encodec_model,
    clip_tokenizer=clip_tokenizer,
    clip_text_model=clip_text_model,
    transformer=None,
    scheduler=noise_scheduler,
)


sample_x_0 = torch.arange(0, MAX_AUDIO_CODES_LENGTH).reshape(1, -1)
assert sample_x_0.shape == torch.Size([1, MAX_AUDIO_CODES_LENGTH])

log_one_hot_x_0 = index_to_log_onehot(sample_x_0, noise_scheduler.num_embed)

sample = sample_x_0.clone()

for timestep in range(noise_scheduler.num_train_timesteps-1, 0, -1):

    timestep_t = torch.tensor([timestep], dtype=torch.long)

    noisy_samples = noise_scheduler.add_noise(log_one_hot_x_0, timestep_t)

    prev_sample_outputs = noise_scheduler.step(model_output=log_one_hot_x_0, timestep=timestep, sample=noisy_samples, use_oracle_q_posterior=True)

    prev_sample_log_probas = prev_sample_outputs.prev_sample_log_probas
    sample = prev_sample_outputs.prev_sample

    if timestep > 0 and timestep % 25 == 0:
        ideal_prev_sample_log_probas = noise_scheduler.q_forward(log_one_hot_x_0, timestep_t-1)

        print("ideal     log_p\n", ideal_prev_sample_log_probas)
        print("predicted log_p\n", prev_sample_log_probas)

        kl_loss = multinomial_kl(ideal_prev_sample_log_probas, prev_sample_log_probas)
        print(timestep, "kl_loss", kl_loss.mean())


print("new_sample", sample)
