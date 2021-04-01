from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np
import copy
import time
import os
import random
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

random.shuffle(league)
league_size = len(league) # Number of policies.  Must be even because we don't want byes or anything like that.
teams_per_member=5 # Number of teams per policy.
assert league_size%2 == 0, "Number of policies in the TEST_LEAGUE_AGENTS folder must be even"

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = (league_size*teams_per_member)//2
random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, reward_function = win_loss_reward_function, home_team = random_configs, away_team = random_configs)

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")


    # I need teams_per_member indexes per policy from 0 to n_agents-1

    matchups=[[] for i in range(league_size)] # league_size x teams_per_member at the end of it, we append to the second dimension over time hence why it is 0 rn
    lookDistance=league_size//4
    if lookDistance==0:
        lookDistance=1

    league.sort(key=lambda x: x.ELO) # Currently sort our league based on ELO

    observation = [[] for i in range(league_size)]
    action = [[] for i in range(league_size)]
    reward = [[] for i in range(league_size)]
    states = [None for i in range(league_size)]



    for i in range(league_size): # Final model is scheduled by the end of the algo
        if league_size-i <= lookDistance: # Protecting us from going OOB
            curDist=league_size-i-1
        else:
            curDist=lookDistance
        available=list(range(1, curDist+1))
        j=0
        while j<teams_per_member: # Just trust me
            if len(matchups[i]) == teams_per_member: # This means that our policy is fully scheduled
                break
            randIndex=-1
            found = False
            newIndex=-1
            otherIndex=-1
            while True: # Emulating a do-while loop for this
                if len(available) > 0:
                    randIndex=random.choice(available) # Generating the matchup value
                else: # Super convoluted edge case fix
                    for c in range(1, i+1): # Trace back the matchup list until we find a matchup that doesn't involve us 
                        for z in range(teams_per_member):
                            if matchups[i-c][z] != i: # If the matchup in question doesn't involvue us, stick the left out policy in the middle of it
                                far_pos = matchups[matchups[i-c][z]].index(i-c) # Searching the far policy for where the matchup is stored, then replacing it with the left out policy
                                matchups[matchups[i-c][z]][far_pos]=i
                                newIndex=i-c
                                otherIndex=matchups[i-c][z]
                                matchups[i-c][z]=i
                                found=True
                                break
                        if found:
                            break

                if found:
                    break
                
                newIndex=i+randIndex

                if len(matchups[newIndex]) < teams_per_member: # If the selected policy has more matches to be made, we are fine
                    break
                else:
                    available.remove(randIndex) # Remove the fully matched policy from consideration for this iteration so we don't choose it again

            # Here we should have a valid match and now we must store it.

            if found:
                matchups[i].append(newIndex)
                matchups[i].append(otherIndex)
                j+=1
            else:
                matchups[i].append(newIndex)
                matchups[newIndex].append(i)
            j+=1

    #print(matchups)
    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5)) # Array of 0's that is the right size
            agent_mappings=[[] for i in range(league_size)] # Will eventually be league_size x teams_per_member x 3
            home_counter=0
            away_counter=env.n_agents//2
            for i in np.arange(league_size): # We must convert from matchup to agent mapping
                for j in np.arange(teams_per_member):
                    if matchups[i][j]>i: # We only create with a home game because away games are created with them
                        agent_mappings[i].append([home_counter, home_counter+1, home_counter+2])
                        home_counter+=3
                        agent_mappings[matchups[i][j]].append([away_counter, away_counter+1, away_counter+2])
                        away_counter+=3




            for i in range(league_size): #action_n refers to an agent per index, so we assign agents sequentially to each policy based on our order config and then update
                flat=[agent for agents in agent_mappings[i] for agent in agents]
                action_n[flat], states[i] = league[i].get_action(observation_n[flat], states[i]) # league is where the policies are stored

            observation_n, reward_n, done_n, _ = env.step(action_n)
            if all(done_n): # We have to update ELO here, redo this calculation for the new system
                old_elo=[x.ELO for x in league] # We need to save the elos until the end
                for i in range(league_size):
                    for j in np.arange(teams_per_member):
                        policy2=matchups[i][j]
                        
                        policyScore=round(sum(reward_n[agent_mappings[i][j]])) # 0 is a loss, .5 is a tie, 1 is a win

                        policyProb=1.0/(1.0+pow(10, (old_elo[i]-old_elo[policy2])/400))

                        league[i].ELO+=k*(policyScore-policyProb)
                    print(str(i)+": "+str(league[i].ELO))
                break

env.close()
