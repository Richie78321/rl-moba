from gym_derk import DerkAgentServer, DerkSession, DerkAppInstance
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
import asyncio
from derk_PPO_LSTM import lstm_agent
from reward_functions import *

device = "cuda:0"
ITERATIONS = 100
agent = lstm_agent(512, device)

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = 20
save_model_every = 100
eval_against_gap = 100
past_models = []

model_checkpoint_schedule = [int(i ** 1.5) for i in range(1000)]
save_folder = "checkpoints/PPO-LSTM-" + str(time.time())
os.mkdir(save_folder)

async def run(env: DerkSession, app: DerkAppInstance):
    for iteration in range(ITERATIONS):
        print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

        if iteration % save_model_every == 0:
            past_models.append(copy.deepcopy(agent))

        if iteration in model_checkpoint_schedule:
            torch.save(agent.state_dict(), save_folder + "/" + str(iteration))
        
        new_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
        await app.update_away_team_config(new_configs)
        await app.update_home_team_config(new_configs)
        observation = []
        done = []
        action = []
        reward = []
        state = None
        observation_n = await env.reset()
        while True:
            action_n, state = agent.get_action(observation_n, state)

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = await env.step(action_n)

            #collect experience data to learn from
            observation.append(observation_n)
            reward.append(reward_n)
            done.append(done_n)
            action.append(action_n)

            if all(done_n):
                break

        # reshapes all collected data to [episode num, timestep]
        observation = np.swapaxes(np.array(observation), 0, 1)
        reward = np.swapaxes(np.array(reward), 0, 1)
        done = np.swapaxes(np.array(done), 0, 1)
        action = np.swapaxes(np.array(action), 0, 1)

        #learn from experience
        print("Training with PPO")
        agent.update(observation, action, reward)

        if iteration % 2 == 0:
            testing_against = max(0, (iteration - eval_against_gap) // save_model_every)
        if iteration % 4 == 0:
            testing_against = 0
        print("\nEvaluating against iteration ", testing_against * save_model_every)

        curr_agent_state = None
        past_agent_state = None

        observation_n = await env.reset()
        while True:
            #get actions for agent (first half of observations) and random agent (second half of observations)
            agent_action_n, curr_agent_state = agent.get_action(observation_n[:env.n_agents//2], curr_agent_state)
            past_action_n, past_agent_state = past_models[testing_against].get_action(observation_n[env.n_agents//2:], past_agent_state)
            action_n = agent_action_n.tolist() + past_action_n.tolist()

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = await env.step(action_n)

            if all(done_n):
              total_home_reward = 0
              total_away_reward = 0
              for i in range(len(env.team_stats)//2):
                total_home_reward += env.team_stats[i][0]
                total_away_reward += env.team_stats[i][1]
              total_home_reward /= (len(env.team_stats)//2)
              total_away_reward /= (len(env.team_stats)//2)

              print("Agent avg reward:", total_home_reward, " Iteration", testing_against * save_model_every ,"reward:", total_away_reward)
              break

async def main():
    random_start = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
    app = DerkAppInstance()
    server = DerkAgentServer(run,args={"app": app}, port=8789,host="127.0.0.1")
    await server.start()
    await app.start()
    await app.run_session(
        n_arenas=n_arenas,
        turbo_mode=True,
        reward_function=win_loss_reward_function,
        home_team=random_start,
        away_team=random_start,
        agent_hosts=[{"uri": server.uri, "regions":[{"sides":"both"}]}]
    )


asyncio.get_event_loop().run_until_complete(main())