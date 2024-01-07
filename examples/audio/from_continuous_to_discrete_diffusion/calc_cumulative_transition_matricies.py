import torch
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm

NUM_VECTORS_IN_CODEBOOK = 1024
NUM_TRAIN_TIMESTEPS = 100

def debug_tensor(name, tens):
    print(f"{name} [{tens.shape}] ({tens.dtype}) is nan", tens.isnan().any().item(), "min max", tens.min().item(), tens.max().item())

def calc_q_transitioning(norm_dim=1):
    Q_transitioning = torch.load('Q_transitioning_raw.pth')

    Q_transitioning_normed = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])
    Q_transitioning_cumulative_normed = torch.zeros([ NUM_TRAIN_TIMESTEPS, NUM_VECTORS_IN_CODEBOOK, NUM_VECTORS_IN_CODEBOOK ])

    for i in tqdm(range(NUM_TRAIN_TIMESTEPS), total=NUM_TRAIN_TIMESTEPS):

        Q_transitioning_current: torch.Tensor = Q_transitioning[i]
        Q_transitioning_current_normed = (Q_transitioning_current + 1e-20) / (Q_transitioning_current.sum(dim=norm_dim, keepdim=True) + 1e-20)
        # could be row of ones for originaly zeroed row
        Q_transitioning_current_normed = Q_transitioning_current_normed / Q_transitioning_current_normed.sum(dim=norm_dim, keepdim=True)

        if Q_transitioning_current.isnan().any():
            raise ValueError(f"iter={i} Q_transitioning_current contains nan")

        if i == 0:
            # init
            Q_transitioning_cummulative_current: torch.Tensor = Q_transitioning_current_normed
        else:
            # prev mult current
            debug_tensor("Q_transitioning_current", Q_transitioning_current)
            debug_tensor("Q_transitioning_cummulative_current", Q_transitioning_cummulative_current)
            Q_transitioning_cummulative_current = Q_transitioning_cummulative_current @ Q_transitioning_current_normed
            Q_transitioning_cummulative_current = (Q_transitioning_cummulative_current + 1e-20) / (Q_transitioning_cummulative_current.sum(dim=norm_dim, keepdim=True) + 1e-20)
            # could be row of ones for originaly zeroed row
            Q_transitioning_cummulative_current = Q_transitioning_cummulative_current / Q_transitioning_cummulative_current.sum(dim=norm_dim, keepdim=True)

        if Q_transitioning_cummulative_current.isnan().any():
            raise ValueError(f"iter={i} Q_transitioning_cummulative_current contains nan")

        # Q_transitioning_cummulative_current
        # todo есть проблема! матрица несеммитричная

        Q_transitioning_current_normed_sum = Q_transitioning_current_normed.sum(dim=norm_dim)
        Q_transitioning_current_normed_dim_1_sum = torch.isclose(Q_transitioning_current_normed_sum, torch.ones_like(Q_transitioning_current_normed_sum))
        if not Q_transitioning_current_normed_dim_1_sum.all():
            raise ValueError(f"Bad norm at dim={norm_dim} for Q_transitioning_normed:", "count", Q_transitioning_current_normed_dim_1_sum.sum(), "values:", Q_transitioning_current_normed_sum[Q_transitioning_current_normed_dim_1_sum], "indexes", Q_transitioning_current_normed_dim_1_sum.nonzero())

        Q_transitioning_cummulative_current_sum = Q_transitioning_cummulative_current.sum(dim=norm_dim)
        Q_transitioning_cummulative_current_dim_1_sum = torch.isclose(Q_transitioning_cummulative_current_sum, torch.ones_like(Q_transitioning_cummulative_current_sum))
        if not Q_transitioning_cummulative_current_dim_1_sum.all():
            raise ValueError(f"Bad norm at dim={norm_dim} for Q_transitioning_cumulative_normed:", "values:", Q_transitioning_cummulative_current_sum[Q_transitioning_cummulative_current_dim_1_sum], "indexes", Q_transitioning_cummulative_current_dim_1_sum.nonzero())

        Q_transitioning_normed[i] = Q_transitioning_current_normed
        Q_transitioning_cumulative_normed[i] = Q_transitioning_cummulative_current


    if Q_transitioning_normed.isnan().any():
        raise ValueError("Q_transitioning_normed contains nans")

    if Q_transitioning_cumulative_normed.isnan().any():
        raise ValueError("Q_transitioning_cumulative_normed contains nans")



    return Q_transitioning_normed, Q_transitioning_cumulative_normed

if __name__ == '__main__':

    Q_transitioning_normed, Q_transitioning_cumulative_normed = calc_q_transitioning(norm_dim=1)

    torch.save(Q_transitioning_normed, 'Q_transitioning_normed.pth')
    torch.save(Q_transitioning_cumulative_normed, 'Q_transitioning_cumulative_norm.pth')

    Q_transitioning_transposed_normed, Q_transitioning_cumulative_transposed_normed = calc_q_transitioning(norm_dim=0)

    torch.save(Q_transitioning_transposed_normed, 'Q_transitioning_transposed_normed.pth')
    torch.save(Q_transitioning_cumulative_transposed_normed, 'Q_transitioning_cumulative_transposed_normed.pth')
