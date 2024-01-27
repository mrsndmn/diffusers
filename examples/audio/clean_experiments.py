
from os import listdir
from os.path import isdir
import shutil

import os.path as path

valid_experiments = set([
    "q_posterior_official_repo_no_auxiliary_loss_weight_init_2024-01-07 10:45:28.491259",
    "q_posterior_official_repo_no_auxiliary_loss_timesteps_sampling_importance_2024-01-07 12:39:25.080691",
    "q_posterior_official_repo_no_auxiliary_loss_norm_q_probas_timesteps_importance_sampling_2024-01-07 14:06:55.482224",
    "q_posterior_official_repo_aux_only_2024-01-07 14:37:35.639452",
    "q_posterior_official_repo_no_auxiliary_loss_norm_q_probas_timesteps_importance_sampling_transitioning_matricies_plus_eye_2024-01-07 15:03:29.180029",
    "q_posterior_official_repo_no_auxiliary_loss_norm_q_probas_timesteps_importance_sampling_transitioning_matricies_plus_eye_single_dense_2024-01-07 15:07:53.614559",
    "q_posterior_official_repo_aux_only_timesteps_importance_sampling_transitioning_matricies_plus_eye_2024-01-07 15:36:11.188800",
    "q_posterior_official_repo_aux_only_timesteps_importance_sampling_transitioning_matricies_plus_eye_dummy_q_posterior_2024-01-08 14:01:46.024382",
    "q_posterior_official_repo_aux_only_dummy_q_posterior_2024-01-08 14:52:24.073872",
    "q_posterior_official_repo_aux_only_timesteps_importance_sampling_transitioning_matricies_plus_eye_dummy_q_posterior_fix_2024-01-08 15:06:40.427117",
    "q_posterior_official_repo_aux_only_timesteps_importance_sampling_transitioning_matricies_plus_eye_dummy_q_posterior_fix_force_no_old_loss_but_with_forward_old_loss_2024-01-09 10:37:57.702383",
    "q_posterior_official_repo_aux_only_timesteps_transitioning_matricies_plus_eye_dummy_q_posterior_fix_force_no_old_loss_but_with_forward_old_loss_no_importance_sampling_2024-01-09 10:56:01.850805",
    "q_posterior_official_repo_aux_only_timesteps_importance_sampling_transitioning_matricies_plus_eye_2024-01-07 15:36:11.188800",
    "q_posterior_official_repo_aux_only_timesteps_transitioning_matricies_plus_eye_dummy_q_posterior_fix_force_no_old_loss_but_with_forward_old_loss_no_importance_sampling_2024-01-09 10:56:01.850805",
    "q_posterior_official_repo_aux_only_dummy_q_posterior_no_importance_2024-01-13 18:01:07.166784",
    "q_posterior_official_repo_aux_only_dummy_q_posterior_2024-01-13 18:01:07.164620",
    "q_posterior_official_repo_aux_only_dummy_q_posterior_10_timesteps_2024-01-13 18:26:59.332448",
    "q_posterior_official_repo_aux_only_dummy_q_posterior_300_timesteps_2024-01-13 18:32:53.991047",
    "q_posterior_official_repo_aux_only_dense_importance_sampling_20_timesteps_2024-01-13 19:21:10.973072",
    "q_posterior_official_repo_aux_only_dense_importance_sampling_300_timesteps_2024-01-13 20:35:35.387459",
    "q_posterior_official_repo_aux_only_dense_importance_sampling_20_timesteps_2024-01-13 19:21:10.973072",
    "q_posterior_official_repo_aux_only_dense_importance_sampling_300_timesteps_2024-01-13 20:35:35.387459",
    "q_posterior_official_repo_aux_only_dummy_q_posterior_tmp_aux_ts_imp_sampling_2024-01-14 23:27:20.301529",
])

raise Exception("valid experiments are old")

path_prefix = "ddpm-audio-mnist-128/logs"

existing_dirs = sorted(listdir(path_prefix))

to_delete_count = 0

for existing_dir in existing_dirs:
    full_path = path.join(path_prefix, existing_dir)
    if isdir(full_path) and existing_dir not in valid_experiments:
        print(existing_dir)
        to_delete_count += 1
        # shutil.rmtree(full_path)


print("valid_experiments", len(valid_experiments))
print("existing_dirs", len(existing_dirs))
print("to_delete_count", to_delete_count)

