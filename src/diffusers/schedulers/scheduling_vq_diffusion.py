# Copyright 2023 Microsoft and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin

@dataclass
class VQDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
            Computed sample x_{t-1} of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.LongTensor
    prev_sample_log_probas: torch.LongTensor


def index_to_log_onehot(x: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    """
    Convert batch of vector of class indices into batch of log onehot vectors

    Args:
        x (`torch.LongTensor` of shape `(batch size, vector length)`):
            Batch of class indices

        num_classes (`int`):
            number of classes to be used for the onehot vectors

    Returns:
        `torch.FloatTensor` of shape `(batch size, num classes, vector length)`:
            Log onehot vectors
    """
    x_onehot = F.one_hot(x, num_classes)
    x_onehot = x_onehot.permute(0, 2, 1)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30)).to(x.device)
    return log_x


def gumbel_noised(logits: torch.FloatTensor, uniform_noise=None, generator: Optional[torch.Generator]=None) -> torch.FloatTensor:
    """
    Apply gumbel noise to `logits`
    """
    if uniform_noise is None:
        uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    else:
        uniform = uniform_noise
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    noised = gumbel_noise + logits
    return noised


def alpha_schedules(num_diffusion_timesteps: int, alpha_cum_start=0.99999, alpha_cum_end=0.000009):
    """
    Cumulative and non-cumulative alpha schedules.

    See section 4.1.
    """
    att = (
        np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1) * (alpha_cum_end - alpha_cum_start)
        + alpha_cum_start
    )
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    att = np.concatenate((att[1:], [1]))
    return at, att


def gamma_schedules(num_diffusion_timesteps: int, gamma_cum_start=0.000009, gamma_cum_end=0.99999):
    """
    Cumulative and non-cumulative gamma schedules.

    See section 4.1.
    """
    ctt = (
        np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1) * (gamma_cum_end - gamma_cum_start)
        + gamma_cum_start
    )
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    ctt = np.concatenate((ctt[1:], [0]))
    return ct, ctt

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def multinomial_kl(log_prob1, log_prob2):   # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

class VQDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    A scheduler for vector quantized diffusion.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_vec_classes (`int`):
            The number of classes of the vector embeddings of the latent pixels. Includes the class for the masked
            latent pixel.
        num_train_timesteps (`int`, defaults to 100):
            The number of diffusion steps to train the model.
        alpha_cum_start (`float`, defaults to 0.99999):
            The starting cumulative alpha value.
        alpha_cum_end (`float`, defaults to 0.00009):
            The ending cumulative alpha value.
        gamma_cum_start (`float`, defaults to 0.00009):
            The starting cumulative gamma value.
        gamma_cum_end (`float`, defaults to 0.99999):
            The ending cumulative gamma value.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_vec_classes: int,
        num_train_timesteps: int = 100,
        alpha_cum_start: float = 0.99999,
        alpha_cum_end: float = 0.000009,
        gamma_cum_start: float = 0.000009,
        gamma_cum_end: float = 0.99999,
        device='cpu',
    ):

        self.num_embed = num_vec_classes

        # By convention, the index for the mask class is the last class index
        self.mask_class = self.num_embed - 1
        self.is_masked = True

        self.alpha_cum_start = alpha_cum_start
        self.alpha_cum_end = alpha_cum_end
        self.gamma_cum_start = gamma_cum_start
        self.gamma_cum_end = gamma_cum_end

        at, att = alpha_schedules(num_train_timesteps, alpha_cum_start=alpha_cum_start, alpha_cum_end=alpha_cum_end)
        ct, ctt = gamma_schedules(num_train_timesteps, gamma_cum_start=gamma_cum_start, gamma_cum_end=gamma_cum_end)

        num_non_mask_classes = self.num_embed - 1
        bt = (1 - at - ct) / num_non_mask_classes
        btt = (1 - att - ctt) / num_non_mask_classes

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)

        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        self.log_at = log_at.float().to(device)
        self.log_bt = log_bt.float().to(device)
        self.log_ct = log_ct.float().to(device)
        self.log_cumprod_at = log_cumprod_at.float().to(device)
        self.log_cumprod_bt = log_cumprod_bt.float().to(device)
        self.log_cumprod_ct = log_cumprod_ct.float().to(device)

        self.log_1_min_ct = log_1_min_a(self.log_ct).to(device)
        self.log_1_min_cumprod_ct = log_1_min_a(self.log_cumprod_ct).to(device)

        # setable values
        self.num_inference_steps = None
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy()).to(device)

    def generate_absolute_noise(self, latents_shape, encodec_model=None, bandwidth=None, device='cpu', max_noise=0.1):

        mask_class = self.mask_class
        return torch.full(latents_shape, mask_class).to(device)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps and diffusion process parameters (alpha, beta, gamma) should be moved
                to.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.log_at = self.log_at.to(device)
        self.log_bt = self.log_bt.to(device)
        self.log_ct = self.log_ct.to(device)
        self.log_cumprod_at = self.log_cumprod_at.to(device)
        self.log_cumprod_bt = self.log_cumprod_bt.to(device)
        self.log_cumprod_ct = self.log_cumprod_ct.to(device)

        self.log_1_min_ct = log_1_min_a(self.log_ct)
        self.log_1_min_cumprod_ct = log_1_min_a(self.log_cumprod_ct)

    # q( x_t | x_0 ) -- formula (4) in paper https://arxiv.org/pdf/2111.14822.pdf
    def q_forward(
        self,
        log_one_hot_x_0_probas: torch.LongTensor, # (`torch.LongTensor` of shape `(batch size, num_classes, num latent pixels)`):
        timesteps: torch.IntTensor, # [ batch_size ]
    ):
        assert len(log_one_hot_x_0_probas.shape) == 3, f'shape expected to be: [bs, num_classes, num latent pixels], but goot: {log_one_hot_x_0_probas.shape}'
        assert len(timesteps.shape) == 1, f'only one dim for timesteps: {timesteps}'

        assert timesteps.shape[0] == log_one_hot_x_0_probas.shape[0], 'timestemps batch size doesnt match to sample batch size'

        timesteps = timesteps.clone()
        timesteps[timesteps < 0] = 0
        assert timesteps.min() >= 0, 'min timestep is zero'

        expected_dim_1 = self.num_embed
        assert log_one_hot_x_0_probas.shape[1] == expected_dim_1, f'log_one_hot_x_0_probas.shape[1] expected to be {expected_dim_1}, but got: {log_one_hot_x_0_probas.shape}'

        # t could contain -1 value
        timesteps = (timesteps + (self.num_train_timesteps + 1))%(self.num_train_timesteps + 1)
        # timesteps = timesteps.clone()
        # timesteps[timesteps < 0] = 0

        log_cumprod_at = extract(self.log_cumprod_at, timesteps, log_one_hot_x_0_probas.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, timesteps, log_one_hot_x_0_probas.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, timesteps, log_one_hot_x_0_probas.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, timesteps, log_one_hot_x_0_probas.shape)       # 1-ct~

        # build \bar{Q_t} v(x_0) - formula 8 https://arxiv.org/pdf/2111.14822.pdf
        log_one_hot_x_0_probas_at = log_one_hot_x_0_probas[:,:-1,:]+log_cumprod_at
        log_one_hot_x_0_probas_ct = log_one_hot_x_0_probas[:,-1:,:]+log_1_min_cumprod_ct
        log_probs = torch.cat(
            [
                log_add_exp(log_one_hot_x_0_probas_at, log_cumprod_bt),
                log_add_exp(log_one_hot_x_0_probas_ct, log_cumprod_ct)
            ],
            dim=1
        )

        assert log_probs.shape == log_one_hot_x_0_probas.shape, f"{log_probs.shape} != {log_one_hot_x_0_probas.shape} shape of log_probs expected to be eauals to log_one_hot_x_0_probas shape"

        return log_probs

    def q_forward_one_timestep(self, log_x_t, t):         # q(x_t|x_{t-1})

        # timesteps = (timesteps + (self.num_train_timesteps + 1))%(self.num_train_timesteps + 1)
        t = t.clone()
        t[t < 0] = 0
        assert t.min() >= 0, 'min timestep is greater or equal zero'

        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_sample(self, log_probs, uniform_noise=None):

        noisy_log_probs = gumbel_noised(log_probs, uniform_noise=uniform_noise)
        sample = noisy_log_probs.argmax(dim=1)

        return sample

    # returns Long Tensor on discrete noisy input with dims [ bs, num latent pixels ]
    def add_noise(
        self,
        # with mask probas
        log_one_hot_x_0_probas: torch.LongTensor, # (`torch.LongTensor` of shape `(batch size, num_classes, num latent pixels)`):
        timesteps: torch.IntTensor, # [ batch_size ]

    ) -> torch.LongTensor:

        # def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        #
        # log_x_start can be onehot or not

        # x_{t-1}
        log_probs_x_t = self.q_forward(log_one_hot_x_0_probas, timesteps)

        uniform_noise_x_t = torch.rand(log_probs_x_t.shape, device=log_probs_x_t.device, generator=None)
        noisy_log_probs_x_t = gumbel_noised(log_probs_x_t, uniform_noise=uniform_noise_x_t)
        x_t_sample = noisy_log_probs_x_t.argmax(dim=1)

        assert x_t_sample.shape == torch.Size([log_one_hot_x_0_probas.shape[0], log_one_hot_x_0_probas.shape[-1]]), f"sample.shape={x_t_sample.shape}, log_one_hot_x_0_probas.shape={log_one_hot_x_0_probas.shape}"

        return {
            "sample": x_t_sample,
        }

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        with_gumbel_noised = True,
        no_q_posterior = False,
    ) -> Union[VQDiffusionSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by the reverse transition distribution. See
        [`~VQDiffusionScheduler.q_posterior`] for more details about how the distribution is computer.

        Args:
            log_p_x_0: (`torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.
            t (`torch.long`):
                The timestep that determines which transition matrices are used.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            generator (`torch.Generator`, or `None`):
                A random number generator for the noise applied to `p(x_{t-1} | x_t)` before it is sampled from.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] or
                `tuple`.

        Returns:
            [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        batch_size = model_output.shape[0]
        timestep_t = torch.tensor([timestep], dtype=torch.long, device=model_output.device)
        timestep_t = timestep_t.repeat(batch_size)

        if timestep == 0 or no_q_posterior:
            log_p_x_t_min_1 = model_output
        else:
            log_p_x_t_min_1 = self.q_posterior(model_output, sample, timestep_t)

        if with_gumbel_noised:
            log_p_x_t_min_1 = gumbel_noised(log_p_x_t_min_1, generator)

        x_t_min_1 = log_p_x_t_min_1.argmax(dim=1)

        if not return_dict:
            return (x_t_min_1,)

        return VQDiffusionSchedulerOutput(prev_sample=x_t_min_1, prev_sample_log_probas=log_p_x_t_min_1)

    # def q_posterior_orig(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
    def q_posterior_orig(self, log_p_x_0, x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_train_timesteps

        batch_size = log_p_x_0.size()[0]

        log_onehot_x_t = index_to_log_onehot(x_t, self.num_embed)

        mask = (x_t == self.mask_class).unsqueeze(1)

        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_onehot_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, x_t.shape[-1])

        log_qt = self.q_forward(log_onehot_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_onehot_x_t.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_embed-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        print("log_qt", log_qt.shape)
        print("ct_cumprod_vector", ct_cumprod_vector.shape)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector

        log_qt_one_timestep = self.q_forward_one_timestep(log_onehot_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_p_x_0.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_embed-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector

        q = log_p_x_0[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp

        # for t=0 masking will be made later, here wi will ignore it
        log_EV_xtmin_given_xt_given_xstart = self.q_forward(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_posterior(self, log_p_x_0, x_t, t):
        """
        Calculates the log probabilities for the predicted classes of the image at timestep `t-1`:

        ```
        p(x_{t-1} | x_t) = sum( q(x_t | x_{t-1} ) * q(x_{t-1} | x_0) * p(x_0 | x_t) / q(x_t | x_0) )
        ```

        Args:
            log_p_x_0 (`torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            t (`torch.Long`):
                The timestep that determines which transition matrix is used.

        Returns:
            `torch.FloatTensor` of shape `(batch size, num classes, num latent pixels)`:
                The log probabilities for the predicted classes of the image at timestep `t-1`.
        """

        assert t.min().item() >= 0 and t.max().item() < self.num_train_timesteps, 'timesteps are ok'

        log_onehot_x_t = index_to_log_onehot(x_t, self.num_embed)

        assert log_p_x_0.shape[1] == self.num_embed - 1, f'q_posterior log_p_x_0.shape[1] expected to be equal to {self.num_embed - 1}, but got shape {log_p_x_0.shape}'

        def print_tensor_statistics(tensor_name, tensor):
            print(f"{tensor_name} [{tensor.shape}] [{tensor.device}]: min={tensor.min():.4f}, max={tensor.max():.4f}, median={tensor.median():.4f}, mean={tensor.float().mean():.4f}")

        log_q_x_t_given_x_0 = self.log_Q_t_transitioning_to_known_class(
            t=t, x_t=x_t, log_onehot_x_t=log_onehot_x_t, cumulative=True
        )
        print_tensor_statistics("log_q_x_t_given_x_0", log_q_x_t_given_x_0)

        log_q_x_t_given_x_t_min_1 = self.log_Q_t_transitioning_to_known_class(
            t=t, x_t=x_t, log_onehot_x_t=log_onehot_x_t, cumulative=False
        )
        print_tensor_statistics("log_q_x_t_given_x_t_min_1", log_q_x_t_given_x_t_min_1)

        # p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0)          ...      p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0)
        #               .                    .                                   .
        #               .                            .                           .
        #               .                                      .                 .
        # p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})  ...      p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})
        q = log_p_x_0 - log_q_x_t_given_x_0

        # sum_0 = p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) + ... + p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}), ... ,
        # sum_n = p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) + ... + p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        print_tensor_statistics("q_log_sum_exp", q_log_sum_exp)

        # p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_0          ...      p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_n
        #                        .                             .                                   .
        #                        .                                     .                           .
        #                        .                                               .                 .
        # p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_0  ...      p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_n
        q = q - q_log_sum_exp

        # (p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}          ...      (p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}
        #                                         .                                                .                                              .
        #                                         .                                                        .                                      .
        #                                         .                                                                  .                            .
        # (p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}  ...      (p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}
        # c_cumulative_{t-1}                                                                                 ...      c_cumulative_{t-1}
        q = self.apply_cumulative_transitions(q, t - 1)
        print_tensor_statistics("q apply_cumulative_transitions", q)

        # ((p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_0) * sum_0              ...      ((p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_0) * sum_n
        #                                                            .                                                                 .                                              .
        #                                                            .                                                                         .                                      .
        #                                                            .                                                                                   .                            .
        # ((p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_{k-1}) * sum_0  ...      ((p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_{k-1}) * sum_n
        # c_cumulative_{t-1} * q(x_t | x_{t-1}=C_k) * sum_0                                                                                       ...      c_cumulative_{t-1} * q(x_t | x_{t-1}=C_k) * sum_0
        log_p_x_t_min_1 = q + log_q_x_t_given_x_t_min_1 + q_log_sum_exp

        # For each column, there are two possible cases.
        #
        # Where:
        # - sum(p_n(x_0))) is summing over all classes for x_0
        # - C_i is the class transitioning from (not to be confused with c_t and c_cumulative_t being used for gamma's)
        # - C_j is the class transitioning to
        #
        # 1. x_t is masked i.e. x_t = c_k
        #
        # Simplifying the expression, the column vector is:
        #                                                      .
        #                                                      .
        #                                                      .
        # (c_t / c_cumulative_t) * (a_cumulative_{t-1} * p_n(x_0 = C_i | x_t) + b_cumulative_{t-1} * sum(p_n(x_0)))
        #                                                      .
        #                                                      .
        #                                                      .
        # (c_cumulative_{t-1} / c_cumulative_t) * sum(p_n(x_0))
        #
        # From equation (11) stated in terms of forward probabilities, the last row is trivially verified.
        #
        # For the other rows, we can state the equation as ...
        #
        # (c_t / c_cumulative_t) * [b_cumulative_{t-1} * p(x_0=c_0) + ... + (a_cumulative_{t-1} + b_cumulative_{t-1}) * p(x_0=C_i) + ... + b_cumulative_{k-1} * p(x_0=c_{k-1})]
        #
        # This verifies the other rows.
        #
        # 2. x_t is not masked
        #
        # Simplifying the expression, there are two cases for the rows of the column vector, where C_j = C_i and where C_j != C_i:
        #                                                      .
        #                                                      .
        #                                                      .
        # C_j != C_i:        b_t * ((b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_0) + ... + ((a_cumulative_{t-1} + b_cumulative_{t-1}) / b_cumulative_t) * p_n(x_0 = C_i) + ... + (b_cumulative_{t-1} / (a_cumulative_t + b_cumulative_t)) * p_n(c_0=C_j) + ... + (b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_{k-1}))
        #                                                      .
        #                                                      .
        #                                                      .
        # C_j = C_i: (a_t + b_t) * ((b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_0) + ... + ((a_cumulative_{t-1} + b_cumulative_{t-1}) / (a_cumulative_t + b_cumulative_t)) * p_n(x_0 = C_i = C_j) + ... + (b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_{k-1}))
        #                                                      .
        #                                                      .
        #                                                      .
        # 0
        #
        # The last row is trivially verified. The other rows can be verified by directly expanding equation (11) stated in terms of forward probabilities.
        return torch.clamp(log_p_x_t_min_1, -70, 0)

    def log_Q_t_transitioning_to_known_class(
        self, *, t: torch.int, x_t: torch.LongTensor, log_onehot_x_t: torch.FloatTensor, cumulative: bool
    ):
        """
        Calculates the log probabilities of the rows from the (cumulative or non-cumulative) transition matrix for each
        latent pixel in `x_t`.

        Args:
            t (`torch.Long`):
                The timestep that determines which transition matrix is used.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            log_onehot_x_t (`torch.FloatTensor` of shape `(batch size, num classes, num latent pixels)`):
                The log one-hot vectors of `x_t`.
            cumulative (`bool`):
                If cumulative is `False`, the single step transition matrix `t-1`->`t` is used. If cumulative is
                `True`, the cumulative transition matrix `0`->`t` is used.

        Returns:
            `torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`:
                Each _column_ of the returned matrix is a _row_ of log probabilities of the complete probability
                transition matrix.

                # When cumulative, returns `self.num_classes - 1` rows because the initial latent pixel cannot be
                masked.

                Where:
                - `q_n` is the probability distribution for the forward process of the `n`th latent pixel.
                - C_0 is a class of a latent pixel embedding
                - C_k is the class of the masked latent pixel

                non-cumulative result (omitting logarithms):
                ```
                q_0(x_t | x_{t-1} = C_0) ... q_n(x_t | x_{t-1} = C_0)
                          .      .                     .
                          .               .            .
                          .                      .     .
                q_0(x_t | x_{t-1} = C_k) ... q_n(x_t | x_{t-1} = C_k)
                ```

                cumulative result (omitting logarithms):
                ```
                q_0_cumulative(x_t | x_0 = C_0)    ...  q_n_cumulative(x_t | x_0 = C_0)
                          .               .                          .
                          .                        .                 .
                          .                               .          .
                q_0_cumulative(x_t | x_0 = C_{k-1}) ... q_n_cumulative(x_t | x_0 = C_{k-1})
                ```
        """

        assert log_onehot_x_t.shape[1] == self.num_embed, f'log_Q_t_transitioning_to_known_class: log_onehot_x_t shape[1] expected to be = {self.num_embed}, but got shape {log_onehot_x_t.shape}'

        if cumulative:
            a = self.log_cumprod_at[t]
            b = self.log_cumprod_bt[t]
            c = self.log_cumprod_ct[t]
        else:
            a = self.log_at[t]
            b = self.log_bt[t]
            c = self.log_ct[t]

        if len(t.shape) > 0:
            a = a.unsqueeze(1).unsqueeze(2)
            b = b.unsqueeze(1).unsqueeze(2)
            c = c.unsqueeze(1).unsqueeze(2)

        if not cumulative:
            # The values in the onehot vector can also be used as the logprobs for transitioning
            # from masked latent pixels. If we are not calculating the cumulative transitions,
            # we need to save these vectors to be re-appended to the final matrix so the values
            # aren't overwritten.
            #
            # `P(x_t!=mask|x_{t-1=mask}) = 0` and 0 will be the value of the last row of the onehot vector
            # if x_t is not masked
            #
            # `P(x_t=mask|x_{t-1=mask}) = 1` and 1 will be the value of the last row of the onehot vector
            # if x_t is masked
            log_onehot_x_t_transitioning_from_masked = log_onehot_x_t[:, -1, :].unsqueeze(1)

        # `index_to_log_onehot` will add onehot vectors for masked pixels,
        # so the default one hot matrix has one too many rows. See the doc string
        # for an explanation of the dimensionality of the returned matrix.
        log_onehot_x_t = log_onehot_x_t[:, :-1, :]

        # this is a cheeky trick to produce the transition probabilities using log one-hot vectors.
        #
        # Don't worry about what values this sets in the columns that mark transitions
        # to masked latent pixels. They are overwrote later with the `mask_class_mask`.
        #
        # Looking at the below logspace formula in non-logspace, each value will evaluate to either
        # `1 * a + b = a + b` where `log_Q_t` has the one hot value in the column
        # or
        # `0 * a + b = b` where `log_Q_t` has the 0 values in the column.
        #
        # See equation 7 for more details.
        print("log_onehot_x_t", log_onehot_x_t.shape)
        # print("a", a.shape)
        # print("b", b.shape)
        log_Q_t = (log_onehot_x_t + a).logaddexp(b)

        # The whole column of each masked pixel is `c`
        mask_class_mask = x_t == self.mask_class
        mask_class_mask = mask_class_mask.unsqueeze(1).expand(-1, self.num_embed - 1, -1)

        # print("c", c.shape)
        # print("mask_class_mask", mask_class_mask.shape)

        if len(c.shape) > 0:
            c_mask = c.repeat(1, mask_class_mask.shape[1], mask_class_mask.shape[2])
            # print("c_mask", c_mask.shape)
            log_Q_t.masked_scatter_(mask_class_mask, c_mask)
        else:
            log_Q_t[mask_class_mask] = c

        if not cumulative:
            log_Q_t = torch.cat((log_Q_t, log_onehot_x_t_transitioning_from_masked), dim=1)

        return log_Q_t

    def apply_cumulative_transitions(self, q, t):
        bsz = q.shape[0]
        a = self.log_cumprod_at[t]
        b = self.log_cumprod_bt[t]
        c = self.log_cumprod_ct[t]

        num_latent_pixels = q.shape[2]
        if len(t.shape) > 0:
            a = a.unsqueeze(1).unsqueeze(2)
            b = b.unsqueeze(1).unsqueeze(2)

            c = c.unsqueeze(1).unsqueeze(2)
            c = c.repeat(1, 1, num_latent_pixels)
        else:
            c = c.expand(bsz, 1, num_latent_pixels)

        q = (q + a).logaddexp(b)
        q = torch.cat((q, c), dim=1)

        return q


class VQDiffusionSchedulerDummyQPosterior(VQDiffusionScheduler):
    def q_posterior(self, log_p_x_0, x_t, t):

        batch_size = log_p_x_0.shape[0]
        zeros = torch.zeros(batch_size, 1, 1).type_as(log_p_x_0)
        log_zero_vector = torch.log(zeros+1.0e-30).expand(-1, -1, x_t.shape[-1])
        log_p_x_0 = torch.cat((log_p_x_0, log_zero_vector), dim=1)

        log_p_x_0_noised = super().q_forward(log_p_x_0, t-1)

        log_p_x_0_noised = torch.clamp(log_p_x_0_noised, -70, 0)
        return log_p_x_0_noised

class VQDiffusionDenseScheduler(nn.Module, SchedulerMixin, ConfigMixin):
    # Без маскированных токенов
    # Создается из заранее вычисленных
    # Матриц перехода

    @register_to_config
    def __init__(
        self,
        q_transition_martices: torch.Tensor = None, # [ num_timesteps, num_classes, num_classes ]
        q_transition_cummulative_martices: torch.Tensor = None, # [ num_timesteps, num_classes, num_classes ]
        q_transition_transposed_martices: torch.Tensor = None, # [ num_timesteps, num_classes, num_classes ]
        q_transition_transposed_cummulative_martices: torch.Tensor = None, # [ num_timesteps, num_classes, num_classes ]
        device='cpu',
    ):
        super().__init__()

        # no mask for dense scheduler
        self.mask_class = -1
        self.is_masked = False

        self.num_train_timesteps = q_transition_martices.shape[0]
        self.num_embed = q_transition_martices.shape[1]
        assert q_transition_martices.shape[1] == q_transition_martices.shape[2], 'q_transition_martices are square'

        assert q_transition_martices.shape == q_transition_cummulative_martices.shape, f'q_transitioning matricies shapes are different: {q_transition_martices.shape} != {q_transition_cummulative_martices.shape}'
        assert q_transition_transposed_martices.shape == q_transition_transposed_cummulative_martices.shape, f'q_transitioning matricies shapes are different: {q_transition_transposed_martices.shape} != {q_transition_transposed_cummulative_martices.shape}'
        assert q_transition_martices.shape == q_transition_transposed_martices.shape, f'q_transitioning matricies shapes are different: {q_transition_martices.shape} != {q_transition_transposed_martices.shape}'

        # [ num_timesteps, num_classes, num_classes ]
        self.register_buffer('q_transition_martices', q_transition_martices.float().to(device))
        self.register_buffer('q_transition_cummulative_martices', q_transition_cummulative_martices.float().to(device))

        self.register_buffer('q_transition_transposed_martices', q_transition_transposed_martices.float().to(device))
        self.register_buffer('q_transition_transposed_cummulative_martices', q_transition_transposed_cummulative_martices.float().to(device))

        self.timesteps = torch.from_numpy(np.arange(0, self.num_train_timesteps)[::-1].copy()).to(device)

        return


    def generate_absolute_noise(self, latents_shape, encodec_model=None, bandwidth=None, device='cpu', max_noise=0.1):

        assert bandwidth == 3.0, '[todo] other bandwidths are not supported'

        batch_size, latent_seq_len = latents_shape # [ bs, latent_seq_len ]
        encodec_scale_factor = 100 # approximately
        raw_audio_length = int(latent_seq_len * encodec_scale_factor)

        noise_mean = torch.zeros([ batch_size, 1, raw_audio_length], device=encodec_model.device)
        noise_std = noise_mean.clone() + max_noise
        noise = torch.normal(mean=noise_mean, std=noise_std).to(encodec_model.device)

        encoder_outputs = encodec_model.encode(
            noise,
            padding_mask=None,
            bandwidth=bandwidth,
        )

        audio_codes = encoder_outputs.audio_codes # [ 1, bs, num_quntisers, latent_seq_len ]
        assert audio_codes.shape[0] == 1, f"{audio_codes.shape[0]} == 1"
        assert audio_codes.shape[1] == batch_size, f"{audio_codes.shape[1]} == {batch_size}"
        audio_codes = audio_codes[0] # [bs, num_quntisers, latent_seq_len ]
        audio_codes = audio_codes.permute(0, 2, 1)   # [ bs, seq_len, num_quantizers ]
        audio_codes = audio_codes.reshape(audio_codes.shape[0], -1) # [ bs, reshaped_latent_seq_len_with_extra (due to approximate encodec_scale_factor) ]
        audio_codes = audio_codes[:, :latent_seq_len]

        assert audio_codes.shape == torch.Size(latents_shape), f"{audio_codes.shape} == {torch.Size(latents_shape)}"

        return audio_codes


    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        raise NotImplemented

    def add_noise(
        self,
        # with mask probas
        log_one_hot_x_0_probas: torch.LongTensor, # (`torch.LongTensor` of shape `(batch size, num_classes, num latent pixels)`):
        timesteps: torch.IntTensor, # [ batch_size ]

    ) -> torch.LongTensor:

        # def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        #
        # log_x_start can be onehot or not

        # x_{t-1}
        log_probs_x_t = self.q_forward(log_one_hot_x_0_probas, timesteps)

        uniform_noise_x_t = torch.rand(log_probs_x_t.shape, device=log_probs_x_t.device, generator=None)
        noisy_log_probs_x_t = gumbel_noised(log_probs_x_t, uniform_noise=uniform_noise_x_t)
        x_t_sample = noisy_log_probs_x_t.argmax(dim=1)

        assert x_t_sample.shape == torch.Size([log_one_hot_x_0_probas.shape[0], log_one_hot_x_0_probas.shape[-1]]), f"sample.shape={x_t_sample.shape}, log_one_hot_x_0_probas.shape={log_one_hot_x_0_probas.shape}"

        return {
            "sample": x_t_sample,
        }

    # q( x_t | x_0 ) -- formula (4) in paper https://arxiv.org/pdf/2111.14822.pdf
    def q_forward(
        self,
        log_one_hot_x_0_probas: torch.LongTensor, # (`torch.LongTensor` of shape `(batch size, num_classes, num latent pixels)`):
        timesteps: torch.IntTensor, # [ batch_size ]
    ):
        assert len(log_one_hot_x_0_probas.shape) == 3, f'shape expected to be: [bs, num_classes, num latent pixels], but goot: {log_one_hot_x_0_probas.shape}'
        assert len(timesteps.shape) == 1, f'only one dim for timesteps: {timesteps}'

        assert timesteps.shape[0] == log_one_hot_x_0_probas.shape[0], 'timestemps batch size doesnt match to sample batch size'

        timesteps = timesteps.clone()
        timesteps[timesteps < 0] = 0
        assert timesteps.min() >= 0, 'min timestep is zero'

        expected_dim_1 = self.num_embed
        # log_one_hot_x_0_probas [ bs, num_embed, sequence_length ]
        assert log_one_hot_x_0_probas.shape[1] == expected_dim_1, f'log_one_hot_x_0_probas.shape[1] expected to be {expected_dim_1}, but got: {log_one_hot_x_0_probas.shape}'

        cummulative_matrix = self.q_transition_cummulative_martices[timesteps]
        # cummulative_matricies =

        print("log_one_hot_x_0_probas", log_one_hot_x_0_probas.shape, log_one_hot_x_0_probas.dtype, "log_one_hot_x_0_probas is nan", log_one_hot_x_0_probas.isnan().any(), "log_one_hot_x_0_probas min", log_one_hot_x_0_probas.min(), "log_one_hot_x_0_probas max", log_one_hot_x_0_probas.max())
        one_hot_x_0_probas = torch.exp(log_one_hot_x_0_probas)

        print("cummulative_matrix", cummulative_matrix.shape, cummulative_matrix.dtype, "cummulative_matrix is nan", cummulative_matrix.isnan().any(), "cummulative_matrix min", cummulative_matrix.min(), "cummulative_matrix max", cummulative_matrix.max())
        print("one_hot_x_0_probas", one_hot_x_0_probas.shape, one_hot_x_0_probas.dtype, "one_hot_x_0_probas is nan", one_hot_x_0_probas.isnan().any(), "one_hot_x_0_probas min", one_hot_x_0_probas.min(), "one_hot_x_0_probas max", one_hot_x_0_probas.max())
        log_probs = torch.log(torch.bmm(cummulative_matrix, one_hot_x_0_probas) + 1e-20)
        log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)

        if log_probs.isnan().any():
            raise ValueError("nan after log")

        assert log_probs.shape == log_one_hot_x_0_probas.shape, f"{log_probs.shape} != {log_one_hot_x_0_probas.shape} shape of log_probs expected to be eauals to log_one_hot_x_0_probas shape"

        return torch.clamp(log_probs, -70, 0)

    def q_transposed_forward(
        self,
        log_one_hot_x_t_probas: torch.LongTensor, # (`torch.LongTensor` of shape `(batch size, num_classes, num latent pixels)`):
        timesteps: torch.IntTensor, # [ batch_size ]
    ):
        # получаем не вероятность из x_0 получить распределение x_t,
        # а получаем распределенеие x_0 из которых можно прийти в конкретный x_t
        assert len(log_one_hot_x_t_probas.shape) == 3, f'shape expected to be: [bs, num_classes, num latent pixels], but goot: {log_one_hot_x_t_probas.shape}'
        assert len(timesteps.shape) == 1, f'only one dim for timesteps: {timesteps}'

        assert timesteps.shape[0] == log_one_hot_x_t_probas.shape[0], 'timestemps batch size doesnt match to sample batch size'

        timesteps = timesteps.clone()
        timesteps[timesteps < 0] = 0
        assert timesteps.min() >= 0, 'min timestep is zero'

        print("q_transposed_forward")

        expected_dim_1 = self.num_embed
        # log_one_hot_x_0_probas [ bs, num_embed, sequence_length ]
        assert log_one_hot_x_t_probas.shape[1] == expected_dim_1, f'log_one_hot_x_t_probas.shape[1] expected to be {expected_dim_1}, but got: {log_one_hot_x_t_probas.shape}'

        # each matrix is normed by dim=0
        cummulative_matrix = self.q_transition_transposed_cummulative_martices[timesteps]
        # each matrix is now normed by dim=1
        cummulative_matrix_transposed = cummulative_matrix.permute(0, 2, 1)
        # cummulative_matricies =

        one_hot_x_0_probas = torch.exp(log_one_hot_x_t_probas)

        log_probs = torch.log(torch.bmm(cummulative_matrix_transposed, one_hot_x_0_probas) + 1e-20)
        log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)

        assert log_probs.shape == log_one_hot_x_t_probas.shape, f"{log_probs.shape} != {log_one_hot_x_t_probas.shape} shape of log_probs expected to be eauals to log_one_hot_x_t_probas shape"

        return torch.clamp(log_probs, -70, 0)


    def q_forward_one_timestep(self, log_x_t_probas, timesteps):
        # получаем не вероятность из x_0 получить распределение x_t,
        # а получаем распределенеие x_0 из которых можно прийти в конкретный x_t

        timesteps = timesteps.clone()
        timesteps[timesteps < 0] = 0
        assert timesteps.min() >= 0, 'min timestep is greater or equal zero'

        transition_matrix = self.q_transition_martices[timesteps]

        x_t_probas = torch.exp(log_x_t_probas)
        log_probs = torch.log(torch.bmm(transition_matrix, x_t_probas) + 1e-20)

        assert log_probs.shape == log_x_t_probas.shape, f"{log_probs.shape} != {log_x_t_probas.shape} shape of log_probs expected to be eauals to log_one_hot_x_0_probas shape"

        log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)

        return torch.clamp(log_probs, -70, 0)

    def q_transposed_forward_one_timestep(self, log_x_t_probas, timesteps):
        # timesteps = (timesteps + (self.num_train_timesteps + 1))%(self.num_train_timesteps + 1)
        timesteps = timesteps.clone()
        timesteps[timesteps < 0] = 0
        assert timesteps.min() >= 0, 'min timestep is greater or equal zero'

        print("q_transposed_forward_one_timestep")

        transition_matrix = self.q_transition_transposed_martices[timesteps]
        transition_matrix_transposed = transition_matrix.permute(0, 2, 1)
        x_t_probas = torch.exp(log_x_t_probas)

        log_probs = torch.log(torch.bmm(transition_matrix_transposed, x_t_probas) + 1e-20)

        assert log_probs.shape == log_x_t_probas.shape, f"{log_probs.shape} != {log_x_t_probas.shape} shape of log_probs expected to be eauals to log_one_hot_x_0_probas shape"

        log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)

        return torch.clamp(log_probs, -70, 0)

    def q_posterior(self, log_p_x_0, x_t, t):            # p_theta(x{t-1}|x_t) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_train_timesteps

        log_onehot_x_t = index_to_log_onehot(x_t, self.num_embed)

        def debug_tensor(name, tens):
            print(f"{name} is nan", tens.isnan().any(), "min max", tens.min().item(), tens.max().item())

        print("q_posterior")
        debug_tensor("log_onehot_x_t", log_onehot_x_t)
        log_qt = self.q_transposed_forward(log_onehot_x_t, t) # q(xt|x0)
        debug_tensor("log_qt", log_qt)

        log_qt_one_timestep = self.q_transposed_forward_one_timestep(log_onehot_x_t, t)        # q(xt|xt_1)
        debug_tensor("log_qt_one_timestep", log_qt_one_timestep)

        q = log_p_x_0 - log_qt
        debug_tensor("q", q)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        debug_tensor("q_norm", q)

        # for t=0 masking will be made later, here wi will ignore it
        log_EV_xtmin_given_xt_given_xstart = self.q_forward(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        # normalise
        log_EV_xtmin_given_xt_given_xstart = log_EV_xtmin_given_xt_given_xstart - torch.logsumexp(log_EV_xtmin_given_xt_given_xstart, dim=1, keepdim=True)

        if log_EV_xtmin_given_xt_given_xstart.isnan().any():
            raise ValueError("nan after log")

        debug_tensor("log_EV_xtmin_given_xt_given_xstart", log_EV_xtmin_given_xt_given_xstart)

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        with_gumbel_noised = True,
        return_dict: bool = True,
    ) -> Union[VQDiffusionSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by the reverse transition distribution. See
        [`~VQDiffusionScheduler.q_posterior`] for more details about how the distribution is computer.

        Args:
            log_p_x_0: (`torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.
            t (`torch.long`):
                The timestep that determines which transition matrices are used.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            generator (`torch.Generator`, or `None`):
                A random number generator for the noise applied to `p(x_{t-1} | x_t)` before it is sampled from.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] or
                `tuple`.

        Returns:
            [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        batch_size = model_output.shape[0]
        timestep_t = torch.tensor([timestep], dtype=torch.long, device=model_output.device)
        timestep_t = timestep_t.repeat(batch_size)

        if timestep == 0:
            log_p_x_t_min_1 = model_output
        else:
            log_p_x_t_min_1 = self.q_posterior(model_output, sample, timestep_t)

        if with_gumbel_noised:
            log_p_x_t_min_1 = gumbel_noised(log_p_x_t_min_1, generator)

        x_t_min_1 = log_p_x_t_min_1.argmax(dim=1)

        if not return_dict:
            return (x_t_min_1,)

        return VQDiffusionSchedulerOutput(prev_sample=x_t_min_1, prev_sample_log_probas=log_p_x_t_min_1)


class VQDiffusionDenseDummyQPosteriorScheduler(VQDiffusionDenseScheduler):

    def q_posterior(self, log_p_x_0, x_t, t):            # p_theta(x{t-1}|x_t) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_train_timesteps

        def debug_tensor(name, tens):
            print(f"{name} is nan", tens.isnan().any(), "min max", tens.min().item(), tens.max().item())

        print("q_posterior dummy")

        noisy_x_t_minus_1_again = self.q_forward(log_p_x_0, t-1)

        if noisy_x_t_minus_1_again.isnan().any():
            raise ValueError("nan after dummy q forward")

        debug_tensor("noisy_x_t_minus_1_again", noisy_x_t_minus_1_again)

        return torch.clamp(noisy_x_t_minus_1_again, -70, 0)

class VQDiffusionDenseTrainedScheduler(VQDiffusionDenseScheduler):

    def __init__(self,
        q_transition_martices_path: str,
        q_transition_cummulative_martices_path: str,
        q_transition_transposed_martices_path: str,
        q_transition_transposed_cummulative_martices_path: str,
        device='cpu'
    ):

        q_transition_martices = torch.load(q_transition_martices_path)
        q_transition_cummulative_martices = torch.load(q_transition_cummulative_martices_path)

        q_transition_transposed_martices = torch.load(q_transition_transposed_martices_path)
        q_transition_transposed_cummulative_martices = torch.load(q_transition_transposed_cummulative_martices_path)

        super().__init__(
            q_transition_martices=q_transition_martices,
            q_transition_cummulative_martices=q_transition_cummulative_martices,
            q_transition_transposed_martices=q_transition_transposed_martices,
            q_transition_transposed_cummulative_martices=q_transition_transposed_cummulative_martices,
            device=device,
        )

class VQDiffusionDenseTrainedDummyQPosteriorScheduler(VQDiffusionDenseDummyQPosteriorScheduler):

    def __init__(self,
        q_transition_martices_path: str,
        q_transition_cummulative_martices_path: str,
        q_transition_transposed_martices_path: str,
        q_transition_transposed_cummulative_martices_path: str,
        device='cpu'
    ):

        q_transition_martices = torch.load(q_transition_martices_path)
        q_transition_cummulative_martices = torch.load(q_transition_cummulative_martices_path)

        q_transition_transposed_martices = torch.load(q_transition_transposed_martices_path)
        q_transition_transposed_cummulative_martices = torch.load(q_transition_transposed_cummulative_martices_path)

        super().__init__(
            q_transition_martices=q_transition_martices,
            q_transition_cummulative_martices=q_transition_cummulative_martices,
            q_transition_transposed_martices=q_transition_transposed_martices,
            q_transition_transposed_cummulative_martices=q_transition_transposed_cummulative_martices,
            device=device,
        )
