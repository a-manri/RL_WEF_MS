#%%
from stable_baselines3.common.monitor import Monitor
from environments.EMS_env import MicrogridEnv, NormalisedMG, EvalMG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.base_class import BaseAlgorithm

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import time

isExperimental = True

location = "experimental_logs/ems/" if isExperimental else "logs/ems/"

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))


n_envs = 8
def create_wrapped_env(log_file=None, weights=np.array([1.0, 4.0, 1.0])):
    env = MicrogridEnv(weights=weights)
    env = NormalisedMG(env)
    if log_file is not None:
        env = Monitor(env, log_file)
    return env

def create_eval_env(log_file=None, weights=None):
    if weights is not None:
        env = NormalisedMG(MicrogridEnv(mode="eval"))
    else:
        env = NormalisedMG(MicrogridEnv(weights=weights, mode="eval"))
    eval_env = EvalMG(env)
    if log_file is not None:
        eval_env = Monitor(eval_env, log_file)
    return eval_env

# Create the vectorized environment
episode_length = 144*4

def create_callback(alg_name, environment, idx):
    return EvalCallback(environment, 
                        best_model_save_path=f'{location}weights_{idx}/{alg_name}',
                        log_path=f'{location}weights_{idx}/{alg_name}', 
                        eval_freq=4*episode_length*16, 
                        deterministic=True, render=False)

set_of_weights = np.array([[1., 4.0, 1.],
                           [1., 3.0, 1.],
                           [1., 2.0, 1.],
                           [1., 1., 1.],
                           [1., 1., 2.],
                           [1., 1., 3.],
                           [1., 1., 4.], 
                           [1., 2., 2.], 
                           [1., 4., 4.]])

weights_dict = {index: weight for index, weight in enumerate(set_of_weights)}

train = True
if train:
    for idx, weights in enumerate(set_of_weights):

        path = f"logs/ems/weights_{idx}/"

        print(f"Training with weights {weights}")

        sac_vec_env = make_vec_env(lambda: create_wrapped_env(weights=weights), n_envs=n_envs, seed=0)
        td3_vec_env = make_vec_env(lambda: create_wrapped_env(weights=weights), n_envs=n_envs, seed=0)
        ppo_vec_env = make_vec_env(lambda: create_wrapped_env(weights=weights), n_envs=n_envs, seed=0)

        sac_env = create_wrapped_env(path + "sac/sac_monitor.csv", weights=weights)
        td3_env = create_wrapped_env(path + "td3/td3_monitor.csv", weights=weights)
        ppo_env = create_wrapped_env(path + "ppo/ppo_monitor.csv", weights=weights)

        sac_model = SAC("MlpPolicy", sac_env, verbose=1, 
                        train_freq=10, batch_size=512, device="cuda")
        td3_model = TD3("MlpPolicy", td3_env, action_noise=action_noise, 
                        verbose=1, train_freq=10, batch_size=512, target_policy_noise=0.1, device="cuda")
        ppo_model = PPO("MlpPolicy", ppo_env, verbose=1, 
                        batch_size=episode_length*n_envs, 
                        device="cuda", n_steps=episode_length*n_envs*2, 
                        n_epochs=12,
                        learning_rate=3e-4, ent_coef=0.1, clip_range=0.12)

        models = [sac_model, td3_model, ppo_model]
        envs = [sac_env, td3_env, ppo_env]
        models_name = ["sac", "td3", "ppo"]
        for model, name, env in zip(models, models_name, envs):
            start_time = time.time()
            model.learn(total_timesteps=1_600_000, callback=create_callback(name, env, idx))
            end_time = time.time()

            print(f"Training {name} took {(end_time - start_time)} seconds")

#%% Plotting the training curves


fig, axs = plt.subplots(3,3, figsize = (10, 6), sharex='col', sharey='row')
#axs = axs.flatten()
for idx, weights in enumerate(set_of_weights):
    path = f"{location}weights_{idx}/"
    for name in ["sac", "td3", "ppo"]:
        ax = axs[np.mod(idx, 3), idx//3]
        training_data = np.load(f"{path}{name}/evaluations.npz")
        scores = training_data["results"].mean(axis = 1)
        std_scores = training_data["results"].std(axis = 1)
        tt = training_data["timesteps"]
        ax.plot(tt, scores, label=name.upper())
        ax.fill_between(tt, scores - std_scores, scores + std_scores, alpha=0.2)
        title_1 = r"$\bar{\lambda}_1 = " + f"{weights[0]} \quad$"
        title_2 = r"$\bar{\lambda}_2 = " + f"{weights[1]}$"
        ax.set_title(title_1 + title_2)
        ax.set_ylim(-50, 850)
        ax.legend(fontsize=8)

        if (idx == 1 or idx == 2 or idx == 0):
            ax.set_ylabel("Mean Reward", fontsize=12)
        if (idx == 4 or idx == 5 or idx == 6):
            ax.set_xlabel("Timesteps", fontsize=12)
fig.tight_layout()
fig.savefig(f"{location}training_curves.png", dpi=300)




# %%
