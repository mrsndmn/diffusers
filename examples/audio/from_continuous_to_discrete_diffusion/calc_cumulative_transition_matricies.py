import torch
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm


Q_transitioning = torch.load('Q_transitioning_raw.pth')

NUM_VECTORS_IN_CODEBOOK = 1024
NUM_TRAIN_TIMESTEPS = 100

Q_transitioning_cumulative = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])

for i in tqdm(range(NUM_TRAIN_TIMESTEPS), total=NUM_TRAIN_TIMESTEPS):

    if i == 0:
        Q_transitioning_current = Q_transitioning[0]
    else:
        Q_transitioning_current = Q_transitioning_current @ Q_transitioning[i]

    Q_transitioning_cumulative[i] = Q_transitioning_current

torch.save(Q_transitioning, 'Q_transitioning_cumulative_raw.pth')
