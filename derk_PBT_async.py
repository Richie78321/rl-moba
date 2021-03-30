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
from PBT import *
from derk_PPO_LSTM import lstm_agent
from reward_functions import *

device = "cuda:0"
ITERATIONS = 1000000
agent = lstm_agent(512, device)

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = 80

# PBT Parameters
population_size = 20
pbt_min_iterations = 10
# Define which hyperparameters to exploit and how to copy the values.
exploit_methods = {
    'learning_rate': lambda x: x,
    'discrete_entropy_coeff': lambda x: x,
    'continuous_entropy_coeff': lambda x: x,
    'value_coeff': lambda x: x,
    'fragments_per_batch': lambda x: x,
}
# Define which hyperparameters to explore and how to.
perturb_explore = get_perturb_explore()
discrete_perturb_explore = get_discrete_perturb_explore()

explore_methods = {
    'learning_rate': perturb_explore,
    'discrete_entropy_coeff': perturb_explore,
    'continuous_entropy_coeff': perturb_explore,
    'value_coeff': perturb_explore,
    'fragments_per_batch': discrete_perturb_explore,
}

# Initialize population with uniformly distributed hyperparameters.
population = [lstm_agent(512, device, hyperparams={
    'learning_rate': np.random.uniform(5e-5, 5e-2),
    'discrete_entropy_coeff': np.random.uniform(0.001, 1.0),
    'continuous_entropy_coeff': np.random.uniform(0.0001, 0.1),
    'value_coeff': np.random.uniform(0.1, 1.5),
    'fragments_per_batch': np.random.randint(low = 10, high = 200),
}) for i in range(population_size)]
# Record the last PBT update
last_PBT_update = [0] * len(population)

model_checkpoint_schedule = [2*int(i ** 1.6) for i in range(1000)]
save_folder = "checkpoints/PPO-LSTM-PBT" + str(time.time())
os.makedirs(save_folder)

hyperparam_history_save_file = os.path.join(save_folder, 'hyperparam_history.json')

# Record the initial hyperparameter configuration of all agents.
update_hyperparameter_history(hyperparam_history_save_file, list(zip(population, [True] * len(population))), 0)

async def run(env: DerkSession, app: DerkAppInstance):
    teams_per_member = (env.n_agents // 3) // population_size
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

        new_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
        await app.update_away_team_config(new_configs)
        await app.update_home_team_config(new_configs)
        
        observation = [[] for i in range(population_size)]
        action = [[] for i in range(population_size)]
        reward = [[] for i in range(population_size)]
        states = [None for i in range(population_size)]

        observation_n = await env.reset()
        with torch.no_grad():
            while True:
                action_n = np.zeros((env.n_agents, 5))
                for i in range(population_size):
                    action_n[league_agent_mappings[i]], states[i] = population[i].get_action(observation_n[league_agent_mappings[i]], states[i])

                #act in environment and observe the new obervation and reward (done tells you if episode is over)
                observation_n, reward_n, done_n, _ = await env.step(action_n)

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