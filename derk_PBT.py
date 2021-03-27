from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np
import copy
import time
import os
from tqdm import tqdm
from torch_truncnorm.TruncatedNormal import TruncatedNormal
import random
from PBT import *
from derk_PPO_LSTM import lstm_agent

torch.autograd.set_detect_anomaly(True)

shaped_reward_function = {
    "damageEnemyStatue":  0.01,
    "damageEnemyUnit":  0.01,
    "killEnemyStatue": 0.1,
    "killEnemyUnit":  0.02,
    "healFriendlyStatue":  0.02,
    "healTeammate1": 0.01,
    "healTeammate2":  0.01,
    "timeSpentHomeBase": 0,
    "timeSpentHomeTerritory": 0,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 0,
    "damageTaken": - 0.01,
    "friendlyFire": - 0.01,
    "healEnemy": - 0.01,
    "fallDamageTaken": -.1,
    "statueDamageTaken": - 0.04,
    "manualBonus": 0,
    "victory": 2,
    "loss": -2,
    "tie": 0,
    "teamSpirit": 0.3,
    "timeScaling": 1.0,
}

win_loss_reward_function = {
    "damageEnemyStatue": 0,
    "damageEnemyUnit": 0,
    "killEnemyStatue": 0,
    "killEnemyUnit": 0,
    "healFriendlyStatue": 0.,
    "healTeammate1": 0,
    "healTeammate2": 0,
    "timeSpentHomeBase": 0,
    "timeSpentHomeTerritory": 0,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 0,
    "damageTaken": 0,
    "friendlyFire": 0,
    "healEnemy": 0,
    "fallDamageTaken": 0,
    "statueDamageTaken": 0,
    "manualBonus": 0,
    "victory": 0.3333333333333,
    "loss": 0,
    "tie": 0.16666666666666,
    "teamSpirit": 1.0,
    "timeScaling": 1.0,
}

classes_team_config = [
      { 'primaryColor': '#ff00ff', 'slots': ['Talons', 'FrogLegs', 'ParalyzingDart'] },
      { 'primaryColor': '#00ff00', 'slots': ['Magnum', 'Trombone', 'VampireGland'] },
      { 'primaryColor': '#ff0000', 'slots': ['Cripplers', 'IronBubblegum', 'HealingGland'] }]

device = "cuda:0"
ITERATIONS = 1000000
agent = lstm_agent(512, device)

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = 80
random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, reward_function = win_loss_reward_function, home_team = random_configs, away_team = random_configs)

# PBT Parameters
population_size = 10
pbt_min_iterations = 10
# Define which hyperparameters to exploit and how to copy the values.
exploit_methods = {
    'learning_rate': lambda x: x,
    'entropy_coeff': lambda x: x,
    'value_coeff': lambda x: x,
}
# Define which hyperparameters to explore and how to.
perturb_explore = get_perturb_explore()
explore_methods = {
    'learning_rate': perturb_explore,
    'entropy_coeff': perturb_explore,
    'value_coeff': perturb_explore,
}

teams_per_member = (env.n_agents // 3) // population_size
# Initialize population with uniformly distributed hyperparameters.
population = [lstm_agent(512, device, hyperparams={
    'learning_rate': np.random.uniform(0.01, 0.1),
    'entropy_coeff': np.random.uniform(0.001, 0.1),
    'value_coeff': np.random.uniform(0.5, 1.5)
}) for i in range(population_size)]
# Record the last PBT update
last_PBT_update = [0] * len(population)

model_checkpoint_schedule = [2*int(i ** 1.6) for i in range(1000)]
save_folder = "checkpoints/PPO-LSTM-PBT" + str(time.time())
os.makedirs(save_folder)

hyperparam_history_save_file = os.path.join(save_folder, 'hyperparam_history.json')

# Record the initial hyperparameter configuration of all agents.
update_hyperparameter_history(hyperparam_history_save_file, list(zip(population, [True] * len(population))), 0)

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

    #randomize matchings between league members
    scrambled_team_IDS = np.random.permutation(env.n_agents // 3)
    league_agent_mappings = []
    for i in range(population_size):
        member_matches = scrambled_team_IDS[teams_per_member*i:teams_per_member*(i+1)]
        league_agent_mappings.append(np.concatenate([(member_matches * 3) + i for i in range(3)], axis = 0))

    if iteration in model_checkpoint_schedule:
        for i in range(population_size):
            torch.save(population[i].state_dict(), save_folder + "/" + str(iteration) + "_" + str(i))

    observation = [[] for i in range(population_size)]
    action = [[] for i in range(population_size)]
    reward = [[] for i in range(population_size)]
    states = [None for i in range(population_size)]

    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5))
            for i in range(population_size):
                action_n[league_agent_mappings[i]], states[i] = population[i].get_action(observation_n[league_agent_mappings[i]], states[i])

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

            #collect experience data to learn from
            for i in range(population_size):
                observation[i].append(observation_n[league_agent_mappings[i]])
                reward[i].append(reward_n[league_agent_mappings[i]])
                action[i].append(action_n[league_agent_mappings[i]])

            if all(done_n):
              break

        # reshapes all collected data to [episode num, timestep]
        for i in range(population_size):
            observation[i] = np.swapaxes(np.array(observation[i]), 0, 1)
            reward[i] = np.swapaxes(np.array(reward[i]), 0, 1)
            action[i] = np.swapaxes(np.array(action[i]), 0, 1)

    #learn from experience
    print("Training League With PPO")
    for i in range(population_size):
        print("\nTraining Population Member", i)
        population[i].update(observation[i], action[i], reward[i])

    agent_pbt_ready = [iteration - x >= pbt_min_iterations - 1 for x in last_PBT_update]
    if any(agent_pbt_ready):
        cumulative_rewards = np.array(reward).sum((1, 2)).tolist()
        print("Cumulative rewards per agent:", cumulative_rewards)

        agents_and_rewards = list(zip(population, cumulative_rewards, agent_pbt_ready))
        updated_agents = pbt_update_bottom(agents_and_rewards, exploit_methods, explore_methods)

        for updated_agent_index in updated_agents:
            last_PBT_update[updated_agent_index] = iteration

        print("Completed PBT update for {} agents".format(len(updated_agents)))

        update_hyperparameter_history(hyperparam_history_save_file, list(zip(population, [x in updated_agents for x in range(len(population))])), iteration)

env.close()
