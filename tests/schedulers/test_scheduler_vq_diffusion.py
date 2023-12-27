import torch
import torch.nn.functional as F

from diffusers import VQDiffusionScheduler
from diffusers.schedulers.scheduling_vq_diffusion import index_to_log_onehot

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

            print("t  ", timesteps)
            print("x_0", x_0)
            print("x_t", x_t)



        return

