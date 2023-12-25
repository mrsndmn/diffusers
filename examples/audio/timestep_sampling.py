import torch
import torch.nn as nn

class TimestepsSampler(nn.Module):

    SAMPLING_STRATEGY_UNIFORM = 'uniform'
    SAMPLING_STRATEGY_IMPORTANCE = 'importance'
    VALID_SAMPLING_STRATEGY = set([SAMPLING_STRATEGY_UNIFORM, SAMPLING_STRATEGY_IMPORTANCE])


    def __init__(self, strategy: str, num_timesteps: int, minimum_history_for_importance_sampling=100):
        super().__init__()

        if strategy not in self.VALID_SAMPLING_STRATEGY:
            raise ValueError("Invalid sampling strategy passed: {strategy}")

        self.strategy = strategy
        self.num_timesteps = num_timesteps

        # used to save device
        self.register_buffer('dummy_device', torch.zeros([0]))

        if strategy == self.SAMPLING_STRATEGY_IMPORTANCE:
            # loss statistic for each timestemp
            self.register_buffer('loss_t_history', torch.zeros(self.num_timesteps))
            self.register_buffer('loss_t_count', torch.zeros(self.num_timesteps))

            self.minimum_history_for_importance_sampling = minimum_history_for_importance_sampling

    def sample_uniform(self, batch_size: int):
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.dummy_device.device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def sample_importance(self, batch_size: int):
        if not (self.loss_t_count > self.minimum_history_for_importance_sampling).all():
            return self.sample_uniform(batch_size)

        Lt_sqrt = torch.sqrt(self.loss_t_history + 1e-10) + 0.0001
        Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
        pt_all = Lt_sqrt / Lt_sqrt.sum()

        t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)

        pt = pt_all.gather(dim=0, index=t)

        return t, pt

    def step(self, kl_loss: torch.Tensor, timesteps: torch.Tensor):
        """
        Params
            kl_loss: shape=[ batch_size ], dtype=torch.long
            timesteps: shape=[ batch_size ], dtype=torch.long
        """

        if self.strategy == self.SAMPLING_STRATEGY_IMPORTANCE:
            Lt2 = kl_loss.pow(2)

            # Calc moving agerage
            Lt2_prev = self.loss_t_history.gather(dim=0, index=timesteps)
            new_loss_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.loss_t_history.scatter_(dim=0, index=timesteps, src=new_loss_history)
            self.loss_t_count.scatter_add_(dim=0, index=timesteps, src=torch.ones_like(Lt2))

        return

    def sample(self, batch_size):
        """
        Returns:
            t: shape=[ batch_size ], dtype=torch.long
                Timesteps
            pt:
                Weight of each timestep
        """

        if self.strategy == self.SAMPLING_STRATEGY_UNIFORM:
            return self.sample_uniform(batch_size)
        elif self.strategy == self.SAMPLING_STRATEGY_IMPORTANCE:
            return self.sample_importance(batch_size)
        else:
            raise ValueError(f"Unhandled sampling strategy: {self.strategy}")

