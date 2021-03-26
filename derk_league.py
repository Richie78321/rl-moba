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

league_size = 10
teams_per_member = (env.n_agents // 3) // league_size
league = [lstm_agent(512, device) for i in range(league_size)]

model_checkpoint_schedule = [2*int(i ** 1.6) for i in range(1000)]
save_folder = "checkpoints/PPO-LSTM-LEAGUE" + str(time.time())
os.makedirs(save_folder)

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

    #randomize matchings between league members
    scrambled_team_IDS = np.random.permutation(env.n_agents // 3)
    league_agent_mappings = []
    for i in range(league_size):
        member_matches = scrambled_team_IDS[teams_per_member*i:teams_per_member*(i+1)]
        league_agent_mappings.append(np.concatenate([(member_matches * 3) + i for i in range(3)], axis = 0))

    if iteration in model_checkpoint_schedule:
        for i in range(league_size):
            torch.save(league[i].state_dict(), save_folder + "/" + str(iteration) + "_" + str(i))

    observation = [[] for i in range(league_size)]
    action = [[] for i in range(league_size)]
    reward = [[] for i in range(league_size)]
    states = [None for i in range(league_size)]

    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5))
            for i in range(league_size):
                action_n[league_agent_mappings[i]], states[i] = league[i].get_action(observation_n[league_agent_mappings[i]], states[i])

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

            #collect experience data to learn from
            for i in range(league_size):
                observation[i].append(observation_n[league_agent_mappings[i]])
                reward[i].append(reward_n[league_agent_mappings[i]])
                action[i].append(action_n[league_agent_mappings[i]])

            if all(done_n):
              break

        # reshapes all collected data to [episode num, timestep]
        for i in range(league_size):
            observation[i] = np.swapaxes(np.array(observation[i]), 0, 1)
            reward[i] = np.swapaxes(np.array(reward[i]), 0, 1)
            action[i] = np.swapaxes(np.array(action[i]), 0, 1)

    #learn from experience
    print("Training League With PPO")
    for i in range(league_size):
        print("\nTraining League Member", i)
        league[i].update(observation[i], action[i], reward[i])

env.close()
