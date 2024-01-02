import torch
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm


Q_transitioning = torch.load('Q_transitioning_raw.pth')

NUM_VECTORS_IN_CODEBOOK = 1024
NUM_TRAIN_TIMESTEPS = 100

Q_transitioning_normed = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])
Q_transitioning_cumulative_normed = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])

for i in tqdm(range(NUM_TRAIN_TIMESTEPS), total=NUM_TRAIN_TIMESTEPS):

    Q_transitioning_current: torch.Tensor = Q_transitioning[i]

    if i == 0:
        Q_transitioning_cummulative_current: torch.Tensor = Q_transitioning_current
    else:
        Q_transitioning_cummulative_current = Q_transitioning_cummulative_current @ Q_transitioning_current

    Q_transitioning_cummulative_current += 1e-3
    # todo есть проблема! матрица несеммитричная
    Q_transitioning_current_normed = Q_transitioning_cummulative_current / Q_transitioning_cummulative_current.sum(dim=1, keepdim=True)

    Q_transitioning_cumulative_normed[i] = Q_transitioning_current_normed
    Q_transitioning_normed[i] = Q_transitioning_current / Q_transitioning_current.sum(dim=1, keepdim=True)

torch.save(Q_transitioning_normed, 'Q_transitioning_normed.pth')
torch.save(Q_transitioning_cumulative_normed, 'Q_transitioning_cumulative_norm.pth')
