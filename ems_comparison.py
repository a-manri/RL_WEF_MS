#%%
import matplotlib
matplotlib.use('TkAgg')
from environments.Cultivates import Cultivates
from environments.SimuEnv import SimuEnv
from environments.WMS_policies import RLIrrigationPolicy
from environments.EMS_policies import RLPumpingPolicy
import numpy as np
from copy import deepcopy
from stable_baselines3 import SAC, TD3, PPO
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
from matplotlib.colors import to_hex
import pickle
import seaborn as sns


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

year = 2017

irrigation_policy = RLIrrigationPolicy(n_crops=1, 
                                       rl_policy=SAC.load("logs/wms/weights_5/sac/best_model.zip"), 
                                       year=year)

seed = 42
set_of_weights = np.array([[1., 4.0, 1.],
                           [1., 3.0, 1.],
                           [1., 2.0, 1.],
                           [1., 1., 1.],
                           [1., 1., 2.],
                           [1., 1., 3.],
                           [1., 1., 4.], 
                           [1., 2., 2.], 
                           [1., 4., 4.]])
#%%
run = False
if run:
    stats = np.zeros((3, len(set_of_weights), 2))  # [n agents, n weights, n metrics]

    for idx, weights in enumerate(set_of_weights):

        path = f"logs/ems/weights_{idx}/"
        sac_agent = RLPumpingPolicy(1, SAC.load(path + "sac/best_model.zip"), True)
        td3_agent = RLPumpingPolicy(1, TD3.load(path + "td3/best_model.zip"), True)
        ppo_agent = RLPumpingPolicy(1, PPO.load(path + "ppo/best_model.zip"), True)
        agents = [sac_agent, td3_agent, ppo_agent]

        for jdex, name, agent in zip([0, 1, 2], ["sac", "td3", "ppo"], agents):
            simu_env = SimuEnv(irrigation_policy=irrigation_policy, 
                            ems_policy=agent,
                            year=year, seed=seed)
            p_day = simu_env.cultivate_env.crops[0].plantation_day
            simu_env.run(init_doy=p_day, total_days=115-30)

            mg_data, crop_data, soil_data = simu_env.get_simu_data()    
            with open(path + f"{name}/" f"simu_data.pkl", "wb") as f:
                pickle.dump({
                    "mg_data": mg_data,
                    "crop_data": crop_data,
                    "soil_data": soil_data
                }, f)
            mg_obs = mg_data["mg_obs"]
            mg_obs_144 = mg_data["end_of_day_samples"]
            v_reqs = crop_data["v_reqs"]
            errors = v_reqs - mg_obs_144[:, 1]  # water requirements vs actual water usage
            smape = np.abs(errors)*2 / (np.abs(v_reqs) + np.abs(mg_obs_144[:, 1]) + 1e-6)
            stats[jdex, idx, 0] = np.mean(smape)*100 # in percentage
            stats[jdex, idx, 1] = np.sum(np.clip(mg_obs[:,5], -np.inf, 0)) 
    np.save("logs/ems/stats.npy", stats) 
# %%
stats = np.zeros((3, len(set_of_weights), 2))
for idx, weights in enumerate(set_of_weights):
    path = f"logs/ems/weights_{idx}/"
    for jdex, name in zip([0, 1, 2], ["sac", "td3", "ppo"]):
            
            simu_data = pickle.load(open(path + f"{name}/" + f"simu_data.pkl", "rb"))
            mg_data = simu_data["mg_data"]
            crop_data = simu_data["crop_data"]
            soil_data = simu_data["soil_data"]
            
            mg_obs = mg_data["mg_obs"]
            mg_obs_144 = mg_data["end_of_day_samples"]
            v_reqs = crop_data["v_reqs"]
            errors = v_reqs - mg_obs_144[:, 1]  # water requirements vs actual water usage
            mape = np.abs(errors) / (np.abs(v_reqs) + 1e-6)
            stats[jdex, idx, 0] = np.mean(mape)*100 # in percentage
            stats[jdex, idx, 1] = np.sum(np.clip(mg_obs[:,5], -np.inf, 0))

#%%
#stats = np.load("logs/ems/stats.npy")
stats[:,:,1] = np.abs(stats[:, :, 1]) # bought energy in kWh

min_val = np.inf
index = None
for j, jtem in enumerate(stats):
    for i, item in enumerate(jtem):
        if item[0] < 5.0:
            if item[1] < min_val:
                min_val = item[1]
                index = (j, i)

algorithms = ["SAC", "TD3", "PPO"]
rows = []
for i, weights in enumerate(set_of_weights):
    row = [r"$\bar{\lambda}_1$ = "+ f"{weights[1]:.2f} / " + r"$\bar{\lambda}_2$ = " + f"{weights[2]:.2f}"]
    for j in range(3):
        error = stats[j, i, 0]
        energy = stats[j, i, 1]
        cell = f"{error:.2f} / {energy:.0f}"
        cell_fmt = cell
        if (j, i) == index:
            cell_fmt = f"\\textbf{{{cell}}}"
        if error < 5.0:
            cell_fmt = f"\\cellcolor{{gray!30}}{cell_fmt}"
        row.append(cell_fmt)
    rows.append(row)

header = ["Weights"] + algorithms
latex_table = tabulate(rows, headers=header, tablefmt="latex_raw", stralign="center", numalign="center")
with open("logs/ems/table.tex", "w") as f:
    f.write(latex_table)

    # Prepare data for heatmap
    error_matrix = stats[:, :, 0]  # shape: (3, n_weights)
    energy_matrix = stats[:, :, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot SMAPE error heatmap
    sns.heatmap(error_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=[f"{w[1]:.1f}/{w[2]:.1f}" for w in set_of_weights],
                yticklabels=algorithms, ax=axes[0], cbar_kws={'label': 'SMAPE (%)'})
    axes[0].set_title("SMAPE Error (%)")
    axes[0].set_xlabel("Weights ($\\bar{\\lambda}_1$/$\\bar{\\lambda}_2$)")
    axes[0].set_ylabel("Algorithm")

    # Plot Energy heatmap
    sns.heatmap(energy_matrix, annot=True, fmt=".0f", cmap="YlOrRd", 
                xticklabels=[f"{w[1]:.1f}/{w[2]:.1f}" for w in set_of_weights],
                yticklabels=algorithms, ax=axes[1], cbar_kws={'label': 'Energy (kWh)'})
    axes[1].set_title("Bought Energy (kWh)")
    axes[1].set_xlabel("Weights ($\\bar{\\lambda}_1$/$\\bar{\\lambda}_2$)")
    axes[1].set_ylabel("Algorithm")

    plt.tight_layout()
    plt.show()

# %%


fig, axs = plt.subplots(3, 3, squeeze=True, figsize=(7, 8), sharey='row')
axs = axs.flatten()

plot_training_curves = True
if plot_training_curves:
    stats = np.zeros((3, len(set_of_weights), 2))  # [n agents, n weights, n metrics]
    for idx, weights in enumerate(set_of_weights):
        path = f"logs/ems/weights_{idx}/"
        
        for jdex, name in zip([0, 1, 2], ["sac", "td3", "ppo"]):
            data = np.load(path + name + "/evaluations.npz")
            results = np.mean(data["results"], axis=1)
            
            axs[idx].plot(results)
        