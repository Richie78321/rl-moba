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
from reward_functions import *

device = "cuda:0"
ITERATIONS = 1000000
agent = lstm_agent(512, device)

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = 800
random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, reward_function = win_loss_reward_function, home_team = random_configs, away_team = random_configs)

# PBT Parameters
population_size = 20
pbt_min_iterations = 10
pbt_fitness_avg_exp = 0.2 # Lower means more focus on present -- 0 for no average
# Define which hyperparameters to exploit and how to copy the values.
exploit_methods = {
    'learning_rate': lambda x: x,
    'discrete_entropy_coeff': lambda x: x,
    'continuous_entropy_coeff': lambda x: x,
    'value_coeff': lambda x: x,
    'minibatch_size': lambda x: x,
    'lstm_fragment_length': lambda x: x,
}
# Define which hyperparameters to explore and how to.
fragment_length_choices = [2,3,5,6,10,15,25,30]
perturb_explore = get_perturb_explore()
discrete_perturb_explore = get_discrete_perturb_explore()
fragment_length_perturb_explore = get_list_explore(fragment_length_choices)

explore_methods = {
    'learning_rate': perturb_explore,
    'discrete_entropy_coeff': perturb_explore,
    'continuous_entropy_coeff': perturb_explore,
    'value_coeff': perturb_explore,
    'minibatch_size': discrete_perturb_explore,
    "lstm_fragment_length": fragment_length_perturb_explore,
}

teams_per_member = (env.n_agents // 3) // population_size
# Initialize population with uniformly distributed hyperparameters.
population = [lstm_agent(512, device, hyperparams={
    'learning_rate': 10 ** np.random.uniform(-5, -2),
    'discrete_entropy_coeff': 10 ** np.random.uniform(-6, -4),
    'continuous_entropy_coeff': 10 ** np.random.uniform(-6, -4),
    'value_coeff': 10 ** np.random.uniform(-1, 0.3),
    'minibatch_size': int(10 ** np.random.uniform(2, 3.5)),
    "lstm_fragment_length": int(random.choice(fragment_length_choices)),
}) for i in range(population_size)]

print(population[0].get_hyperparams())

# Record the last PBT update
last_PBT_update = [0] * len(population)
population_PBT_fitness = np.zeros(len(population), dtype=float)

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
        try:
            population[i].update(observation[i], action[i], reward[i])
        except Exception as e:
            print("\n\nError ocurred training population member number ", i, "!!!!\n", e, "\n\n")

    # Update PBT reward running average
    cumulative_rewards = np.array(reward).sum((1, 2))
    print("Cumulative rewards per agent (this iteration):", cumulative_rewards)

    # Add cumulative rewards to exponential moving average
    population_PBT_fitness = (population_PBT_fitness * pbt_fitness_avg_exp) + (cumulative_rewards * (1 - pbt_fitness_avg_exp))
    print("Moving average of cumulative rewards per agent:", population_PBT_fitness)

    agent_pbt_ready = [iteration - x >= pbt_min_iterations - 1 for x in last_PBT_update]
    if any(agent_pbt_ready):
        agents_and_rewards = list(zip(population, population_PBT_fitness.tolist(), agent_pbt_ready))
        exploiter_and_exploited_indices = pbt_update_bottom(agents_and_rewards, exploit_methods, explore_methods)

        for exploiter_index, exploited_index in exploiter_and_exploited_indices:
            last_PBT_update[exploiter_index] = iteration

            # Copy moving average from exploiter to exploited
            population_PBT_fitness[exploiter_index] = population_PBT_fitness[exploited_index]
            # Can also set the average to zero
            # population_PBT_fitness[exploiter_index] = 0

        print("Completed PBT update for {} agents".format(len(exploiter_and_exploited_indices)))

        if len(exploiter_and_exploited_indices) > 0:
            updated_agents, _ = list(zip(*exploiter_and_exploited_indices))
            update_hyperparameter_history(hyperparam_history_save_file, list(zip(population, [x in updated_agents for x in range(len(population))])), iteration)

env.close()
