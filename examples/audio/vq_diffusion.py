import torch
from diffusers import VQDiffusionPipeline

pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", force_download=False, local_files_only=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device", device)

pipeline = pipeline.to(device)

image = pipeline("student makes diffusion neural networks").images[0]

# save image
image.save("./teddy_bear.png")