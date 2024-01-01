import torch
import torch.nn.functional as F

# from diffusers import
from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot, VQDiffusionDenseScheduler, VQDiffusionDenseUniformMaskScheduler, VQDiffusionScheduler

from .test_schedulers import SchedulerCommonTest


class VQDiffusionSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (VQDiffusionScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_vec_classes": 4097,
            "num_train_timesteps": 100,
        }

        config.update(**kwargs)
        return config

    def dummy_sample(self, num_vec_classes):
        batch_size = 4
        height = 8
        width = 8

        sample = torch.randint(0, num_vec_classes, (batch_size, height * width))

        return sample

    def dummy_sample_probas(self, num_vec_classes):
        batch_size = 4

        height = 8
        width = 8

        sample = torch.rand(size=(batch_size, num_vec_classes, height * width))

        return sample


    @property
    def dummy_sample_deter(self):
        assert False

    def dummy_model(self, num_vec_classes):
        def model(sample, t, *args):
            batch_size, num_latent_pixels = sample.shape
            logits = torch.rand((batch_size, num_vec_classes - 1, num_latent_pixels))
            return_value = F.log_softmax(logits.double(), dim=1).float()
            return return_value

        return model

    def test_timesteps(self):
        for timesteps in [2, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_num_vec_classes(self):
        for num_vec_classes in [5, 100, 1000, 4000]:
            self.check_over_configs(num_vec_classes=num_vec_classes)

    def test_time_indices(self):
        for t in [0, 50, 99]:
            self.check_over_forward(time_step=t)

    def test_add_noise_device(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(100)

            # [ batch_size, num classes, num pixels ]
            log_one_hot_x_0_probas = self.dummy_sample_probas(scheduler.num_embed)
            batch_size = log_one_hot_x_0_probas.shape[0]
            t = scheduler.timesteps[:batch_size]
            noisy_sample = scheduler.add_noise(log_one_hot_x_0_probas, t)

            # [batch_size, num pixels]
            self.assertEqual(noisy_sample.shape, torch.Size([log_one_hot_x_0_probas.shape[0], log_one_hot_x_0_probas.shape[-1]]))

        return

    def test_q_posterior(self):

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(100)

            # [ batch_size, num classes, num pixels ]
            log_one_hot_x_0_probas = self.dummy_sample_probas(scheduler.num_embed)
            batch_size = log_one_hot_x_0_probas.shape[0]

            x_t = torch.randint(0, scheduler.num_embed, size=[batch_size, log_one_hot_x_0_probas.shape[-1]])

            t = scheduler.timesteps[:batch_size]
            print("log_one_hot_x_0_probas", log_one_hot_x_0_probas.shape)
            print("x_t", x_t.shape)
            print("t", t.shape)
            log_x_t_min_1 = scheduler.q_posterior(log_one_hot_x_0_probas[:, :-1, :], x_t=x_t, t=t)

            assert log_x_t_min_1.shape == log_one_hot_x_0_probas.shape

        return

    def test_q_forward(self):

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(100)

            # [ batch_size, num classes, num pixels ]
            log_one_hot_x_0_probas = self.dummy_sample_probas(scheduler.num_embed)
            batch_size = log_one_hot_x_0_probas.shape[0]
            num_latent_codes = log_one_hot_x_0_probas.shape[-1]

            x_0 = torch.randint(0, scheduler.num_embed - 1, size=[batch_size, num_latent_codes])
            log_one_hot_x_0 = index_to_log_onehot(x_0, scheduler.num_embed)

            timesteps = scheduler.timesteps[:batch_size*25:25]
            print("log_one_hot_x_0_probas", log_one_hot_x_0_probas.shape)
            print("x_0", x_0.shape)
            print("t", timesteps.shape)

            log_q_x_t_given_x_0 = scheduler.q_forward(log_one_hot_x_0, timesteps)
            x_t = scheduler.q_sample(log_q_x_t_given_x_0)

            print("t  ", timesteps)
            print("x_0", x_0)
            print("x_t", x_t)

        return

    def test_q_posterior_paper(self):

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(100)

            # [ batch_size, num classes, num pixels ]
            log_one_hot_x_0_probas = self.dummy_sample_probas(scheduler.num_embed)
            batch_size = log_one_hot_x_0_probas.shape[0]
            num_latent_codes = log_one_hot_x_0_probas.shape[-1]

            x_0 = torch.randint(0, scheduler.num_embed - 1, size=[batch_size, num_latent_codes])
            log_one_hot_x_0 = index_to_log_onehot(x_0, scheduler.num_embed)

            timesteps = scheduler.timesteps[:batch_size*25:25]
            print("log_one_hot_x_0_probas", log_one_hot_x_0_probas.shape)
            print("x_0", x_0.shape)
            print("t", timesteps.shape)

            log_q_x_t_given_x_0 = scheduler.q_forward(log_one_hot_x_0, timesteps)
            x_t = scheduler.q_sample(log_q_x_t_given_x_0)

            # scheduler.q_posterior_only(log_one_hot_x_0, x_t)
            raise NotImplemented

        return

    def test_dense_q_transposed(self):
        n_vectors = 10
        n_train_timesteps = 100

        dense_scheduler = VQDiffusionDenseUniformMaskScheduler(
            num_vec_classes=n_vectors,
            num_train_timesteps=n_train_timesteps,
        )


        batch_size = n_train_timesteps
        assert batch_size <= n_train_timesteps
        sequence_length = 128
        timesteps = torch.arange(0, batch_size)
        log_one_hot_x_0_probas = torch.log(torch.zeros([batch_size, n_vectors, sequence_length]) + 1e-20)

        log_one_hot_x_0_probas_noisy = dense_scheduler.add_noise(log_one_hot_x_0_probas, timesteps)
        log_one_hot_x_0_probas_noisy = log_one_hot_x_0_probas_noisy['sample']

        expected_size = torch.Size([batch_size, sequence_length])
        assert expected_size == log_one_hot_x_0_probas_noisy.shape, f"{expected_size} != {log_one_hot_x_0_probas_noisy.shape}"

        assert log_one_hot_x_0_probas_noisy.dtype == torch.long, f"log_one_hot_x_0_probas_noisy.dtype == {log_one_hot_x_0_probas_noisy.dtype}"

        assert log_one_hot_x_0_probas_noisy.min().item() >= 0
        assert log_one_hot_x_0_probas_noisy.max().item() < n_vectors

        return

    def test_dense_q_transposed_q_forward(self):
        n_vectors = 10
        n_train_timesteps = 10

        original_scheduler = VQDiffusionScheduler(n_vectors, n_train_timesteps)

        dense_scheduler = VQDiffusionDenseUniformMaskScheduler(
            num_vec_classes=n_vectors,
            num_train_timesteps=n_train_timesteps,
        )

        batch_size = n_train_timesteps
        assert batch_size <= n_train_timesteps
        sequence_length = n_vectors
        timesteps = torch.arange(0, batch_size)
        eye = torch.eye(sequence_length).unsqueeze(0).repeat(batch_size, 1, 1)
        log_one_hot_x_0_probas = torch.log(eye + 1e-20)
        print("log_one_hot_x_0_probas", log_one_hot_x_0_probas.shape)

        log_one_hot_x_0_probas_orig_forward = original_scheduler.q_forward(log_one_hot_x_0_probas, timesteps)
        log_one_hot_x_0_probas_dense_forward = dense_scheduler.q_forward(log_one_hot_x_0_probas, timesteps)

        # ignore last column!
        one_hot_x_0_probas_orig_forward = torch.exp(log_one_hot_x_0_probas_orig_forward)[:, :-1, :]
        one_hot_x_0_probas_dense_forward = torch.exp(log_one_hot_x_0_probas_dense_forward)[:, :-1, :]

        assert one_hot_x_0_probas_orig_forward.shape == one_hot_x_0_probas_dense_forward.shape, f"{log_one_hot_x_0_probas_orig_forward.shape} == {log_one_hot_x_0_probas_dense_forward.shape}"
        assert torch.allclose(one_hot_x_0_probas_orig_forward, one_hot_x_0_probas_dense_forward, atol=0.1), f'all values for dense vq scheduler are the same as original scheduler'

        return
