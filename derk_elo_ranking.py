from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np
import copy
import time
import os
from random import shuffle
from tqdm import tqdm
from torch_truncnorm.TruncatedNormal import TruncatedNormal
from derk_PPO_LSTM import lstm_agent
from reward_functions import *

"""
Plan

Assign games based on policy vs policy instead of team vs team

Run multiple iterations of each game and pick the winner based on win %

Adjust ELO for each team accordingly

Intelligently assign the next matchup

"""

device = "cuda:0"
ITERATIONS = 1000000
root_dir="checkpoints/TEST_LEAGUE_AGENTS"
league=[]
k=30 #ELO weight parameter
count=0
for root, dirs, files in os.walk(root_dir):
    for name in files: # Load in the agents stored at root_dir
        temp=lstm_agent(512, device)
        temp.load_state_dict(torch.load(os.path.join(root, name)))
        temp.id=count
        temp.elo = 1000
        count+=1
        league.append(temp)

shuffle(league)
league_size = len(league) # Number of policies.  Must be even because we don't want byes or anything like that.
teams_per_member=5 # Number of teams per policy.  Must be odd so there are no ties.  Best of x.
assert league_size%2 == 0, "Number of policies in the TEST_LEAGUE_AGENTS folder must be even"
assert teams_per_member%2 == 1, "Number of teams per policy must be odd"

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = (league_size*teams_per_member)//2
random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, reward_function = win_loss_reward_function, home_team = random_configs, away_team = random_configs)

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

    #randomize matchings between policies to start (This will eventually be more intelligent) - Can just sort by ELO and then assign that way
    #Policy index 0 plays policy index 1, 2 plays 3, and so on.
    #policy_matchup_IDS=np.random.permutation(league_size)

    league.sort(key=lambda x: x.ELO) # Currently sort our league based on ELO

    observation = [[] for i in range(league_size)]
    action = [[] for i in range(league_size)]
    reward = [[] for i in range(league_size)]
    states = [None for i in range(league_size)]

    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5)) # Array of 0's that is the right size
            for i in range(league_size): #action_n refers to an agent per index, so we assign agents sequentially to each policy based on our order config and then update
                action_n[list(range(teams_per_member*i*3, teams_per_member*(i+1)*3))], states[i] = league[i].get_action(observation_n[list(range(teams_per_member*i*3, teams_per_member*(i+1)*3))], states[i]) # league is where the policies are stored
            # All of a policy's agents are stored together, so we just need to sum the reward functions of each policy in the matchup and compare
            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

            if all(done_n): # We have to update ELO here
                for i in range(0, league_size, 2):
                    policy1score=sum(reward_n[teams_per_member*i*3:teams_per_member*(i+1)*3])
                    policy2score=sum(reward_n[teams_per_member*(i+1)*3:teams_per_member*(i+2)*3])

                    policy1prob=1.0/(1.0+pow(10, (league[i].ELO-league[i+1].ELO)/400))
                    policy2prob=1.0/(1.0+pow(10, (league[i+1].ELO-league[i].ELO)/400))

                    policy1actual=0
                    policy2actual=0


                    if policy1score>policy2score:
                        policy1actual=1
                    elif policy1score<policy2score:
                        policy2actual=1
                    else:
                        policy1actual=.5
                        policy2actual=.5


                    league[i].ELO+=k*(policy1actual-policy1prob)
                    league[i+1].ELO+=k*(policy2actual-policy2prob)
                    print("Policy "+str(league[i].id)+" vs Policy "+str(league[i+1].id))
                    print(league[i].ELO)
                    print(league[i+1].ELO)

                break

env.close()
