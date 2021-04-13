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

OBS_KEYS = [
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
        act = torch.Tensor(act).to(self.device).flatten(end_dim = 1) #flatten actions
        obs = torch.Tensor(obs).to(self.device)
        obs.requires_grad = True

        continuous_means, continuous_stds, discrete_output, value, _ = self(obs, None, act)

        normal_dists = torch.distributions.Normal(continuous_means, continuous_stds)
        entropy = normal_dists.entropy()

        for i in range(len(discrete_output)):
            discrete_probs = nn.functional.log_softmax(discrete_output[i], dim=1).exp()
            discrete_dist = torch.distributions.Categorical(discrete_probs)
            entropy = torch.cat((entropy, discrete_dist.entropy().unsqueeze(1)), axis=1)

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

        obs_grad = torch.abs(obs.grad)
        totals = torch.sum(obs_grad, axis = 2)

        percent_obs_grad = (obs_grad / totals.unsqueeze(-1)) * 100
        avg_percent_obs_grad = percent_obs_grad.mean(axis = 0).mean(axis = 0)

        return avg_percent_obs_grad.cpu().detach().numpy().tolist()


league_elos = {
    "0_best": 671.9694965315853,
    "2_best": 681.0078603819308,
    "6_best": 679.0950624180409,
    "18_best": 692.8526928057762,
    "10_best": 706.7188043251175,
    "26_best": 723.353629736221,
    "34_best": 763.1220060621678,
    "44_best": 831.8102654441604,
    "54_best": 845.541286578307,
    "66_best": 994.9128089854157,
    "78_best": 998.9991291825171,
    "92_best": 1027.9154992180763,
    "106_best": 1041.5461901564818,
    "168_best": 1060.8823078875698,
    "152_best": 1078.2139891626362,
    "512_best": 1080.392222486208,
    "120_best": 1086.0525791932307,
    "512_19": 1089.4788755749073,
    "322_best": 1093.2457664500548,
    "412_best": 1101.5239861942139,
    "136_best": 1099.6611801870292,
    "436_best": 1106.5015669631098,
    "390_best": 1101.8799931086185,
    "460_best": 1110.1297327335706,
    "240_best": 1108.4797596004694,
    "222_best": 1116.6452538332658,
    "366_best": 1121.7923736279843,
    "280_best": 1129.9202133491603,
    "300_best": 1131.6277873681545,
    "486_best": 1131.2987233903511,
    "186_best": 1144.2066114425938,
    "344_best": 1139.3215315445293,
    "202_best": 1151.6069512738852,
    "260_best": 1158.293862802676
}

device = "cuda:0"
ITERATIONS = 1000000

league = []
league_analysis = []
root_dir = "checkpoints/PPO-LSTM-PBT-1024"

for name in league_elos:
    temp = analysis_agent(1024, device)
    temp.load_state_dict(torch.load(os.path.join(root_dir, name)))
    temp.name = name
    league.append(temp)

teams_per_member = len(league)

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

max_arenas = 800
teams_per_member = max_arenas // (len(league) // 2)
n_arenas = (len(league)*teams_per_member) // 2

random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, home_team = random_configs, away_team = random_configs)

for i in range(1):
    #randomize matchings between league members
    scrambled_team_IDS = np.random.permutation(env.n_agents // 3)
    league_agent_mappings = []
    for i in range(len(league)):
        member_matches = scrambled_team_IDS[teams_per_member*i:teams_per_member*(i+1)]
        league_agent_mappings.append(np.concatenate([(member_matches * 3) + i for i in range(3)], axis = 0))

    observation = [[] for i in range(len(league))]
    action = [[] for i in range(len(league))]
    reward = [[] for i in range(len(league))]
    states = [None for i in range(len(league))]

    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5))
            for i in range(len(league)):
                action_n[league_agent_mappings[i]], states[i] = league[i].get_action(observation_n[league_agent_mappings[i]], states[i])

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

            #collect experience data to learn from
            for i in range(len(league)):
                observation[i].append(observation_n[league_agent_mappings[i]])
                reward[i].append(reward_n[league_agent_mappings[i]])
                action[i].append(action_n[league_agent_mappings[i]])

            if all(done_n):
              break

    # reshapes all collected data to [episode num, timestep]
    for i in range(len(league)):
        observation[i] = np.swapaxes(np.array(observation[i]), 0, 1)
        reward[i] = np.swapaxes(np.array(reward[i]), 0, 1)
        action[i] = np.swapaxes(np.array(action[i]), 0, 1)

        league_analysis.append(league[i].analyze(observation[i], action[i]))

for i, key in enumerate(OBS_KEYS):
    x = np.zeros(len(league))
    y = np.zeros(len(league))

    for j in range(len(league)):
        x[j] = league_elos[league[j].name]
        y[j] = league_analysis[j][i]

    plt.scatter(x, y)
    plt.title(key + " Percent Gradient vs ELO")
    plt.xlabel("ELO")
    plt.ylabel("Percent of Gradient")
    plt.savefig("analysis/" + key + ".png")
    plt.close()
