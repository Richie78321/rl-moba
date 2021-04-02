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
import matplotlib.pyplot as plt

observation_keys = [
"Hitpoints",
"Ability0Ready",
"FriendStatueDistance",
"FriendStatueAngle",
"Friend1Distance",
"Friend1Angle",
"Friend2Distance",
"Friend2Angle",
"EnemyStatueDistance",
"EnemyStatueAngle",
"Enemy1Distance",
"Enemy1Angle",
"Enemy2Distance",
"Enemy2Angle",
"Enemy3Distance",
"Enemy3Angle",
"HasFocus",
"FocusRelativeRotation",
"FocusFacingUs",
"FocusFocusingBack",
"FocusHitpoints",
"Ability1Ready",
"Ability2Ready",
"FocusDazed",
"FocusCrippled",
"HeightFront1",
"HeightFront5",
"HeightBack2",
"PositionLeftRight",
"PositionUpDown",
"Stuck",
"UnusedSense31",
"HasTalons",
"HasBloodClaws",
"HasCleavers ",
"HasCripplers",
"HasHealingGland",
"HasVampireGland",
"HasFrogLegs",
"HasPistol",
"HasMagnum",
"HasBlaster",
"HasParalyzingDart",
"HasIronBubblegum",
"HasHeliumBubblegum",
"HasShell",
"HasTrombone",
"FocusHasTalons",
"FocusHasBloodClaws",
"FocusHasCleavers",
"FocusHasCripplers",
"FocusHasHealingGland",
"FocusHasVampireGland",
"FocusHasFrogLegs",
"FocusHasPistol",
"FocusHasMagnum",
"FocusHasBlaster",
"FocusHasParalyzingDart",
"FocusHasIronBubblegum",
"FocusHasHeliumBubblegum",
"FocusHasShell",
"FocusHasTrombone",
"UnusedExtraSense30",
"UnusedExtraSense31"]

class analysis_agent(lstm_agent):
    def __init__(self, lstm_size, device, activation = nn.Tanh()):
        super().__init__(lstm_size, device, activation)
        self.device = device

    def analyze(self, obs, act):
        act = torch.Tensor(act).to(self.device)
        obs = torch.Tensor(obs).to(self.device)
        obs.requires_grad = True

        continuous_means, discrete_output, value, _ = self(obs, None, act)

        print(value)

        print(continuous_means)

        normal_dists = torch.distributions.Normal(continuous_means, self.logstd.exp())
        entropy = normal_dists.entropy()

        print(self.logstd.exp())

        for i in range(len(discrete_output)):
            discrete_probs = nn.functional.log_softmax(discrete_output[i], dim=1).exp()
            discrete_dist = torch.distributions.Categorical(discrete_probs)
            entropy = torch.cat((entropy, discrete_dist.entropy().unsqueeze(1)), axis=1)

        print(entropy.mean(axis=0))

        self.optimizer.zero_grad()
        loss = continuous_means.mean()
        #loss = entropy[1]
        loss.mean().backward(retain_graph = True)

        '''
        time_obs_grad = torch.abs(obs.grad).reshape(obs.grad.shape[0] // 150, 150, 64).mean(axis=0)
        print(time_obs_grad.shape)
        mean_time_obs_grad = (time_obs_grad / time_obs_grad.sum(axis=1).unsqueeze(-1)) * 100

        plt.plot(mean_time_obs_grad[:,0].detach().cpu().numpy())
        plt.show()
        '''

        #take absolute value of gradients
        overall_mean_obs_grad = torch.abs(obs.grad).mean(axis=0)
        overall_mean_obs_grad_percentages = (overall_mean_obs_grad / overall_mean_obs_grad.sum()) * 100

        print("\n")
        for i, grad_value in enumerate(overall_mean_obs_grad_percentages.cpu().detach().numpy().tolist()):
            print(observation_keys[i] + ' ' * (24 - len(observation_keys[i])), grad_value)


device = "cuda:0"
ITERATIONS = 1000000
agent = analysis_agent(512, device)
agent.load_state_dict(torch.load("checkpoints/PPO-LSTM-PBT1617169679.9932897/300_18"))

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = 40
random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, home_team = random_configs, away_team = random_configs)

for i in range(92378589027):
    observations = []
    actions = []
    state = None
    observation_n = env.reset()
    while True:
        action_n, state = agent.get_action(torch.Tensor(observation_n).to(device), state)
        observation_n, reward_n, done_n, _ = env.step(action_n)
        observations.append(observation_n)
        actions.append(action_n)

        if all(done_n):
          break

    # reshapes all collected data to [episode num, timestep]
    observations = np.swapaxes(np.array(observations), 0, 1)
    actions = np.swapaxes(np.array(actions), 0, 1)
    #reshape the observation data into one big batch
    observations = observations.reshape(-1, observations.shape[2])
    actions = actions.reshape(-1, actions.shape[2])

    agent.analyze(observations, actions)
