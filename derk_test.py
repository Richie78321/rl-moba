from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np

#custom reward function
reward_function = {
    "damageEnemyStatue": 4,
    "damageEnemyUnit": 2,
    "killEnemyStatue": 4,
    "killEnemyUnit": 2,
    "healFriendlyStatue": 2,
    "healTeammate1": 1,
    "healTeammate2": 1,
    "timeSpentHomeBase": 0,
    "timeSpentHomeTerritory": 0,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 0,
    "damageTaken": -1,
    "friendlyFire": -1,
    "healEnemy": -1,
    "fallDamageTaken": -10,
    "statueDamageTaken": -4,
    "manualBonus": 0,
    "victory": 100,
    "loss": -100,
    "tie": 0,
    "teamSpirit": 0.2,
    "timeScaling": 1.0,
}

class nn_agent(nn.Module):
    def __init__(self, hidden_size, device, activation = nn.Tanh()):
        super().__init__()
        self.device = device

        self.state_processor = nn.Sequential(nn.Linear(64, hidden_size),
                                             activation,
                                             nn.Linear(hidden_size, hidden_size),
                                             activation)

        self.action_discrete = [False, False, False, True, True]

        self.action_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, 2), nn.Tanh()),
                                           nn.Sequential(nn.Linear(hidden_size, 2), nn.Tanh()),
                                           nn.Sequential(nn.Linear(hidden_size, 2), nn.Sigmoid()),
                                           nn.Linear(hidden_size, 4),
                                           nn.Linear(hidden_size, 8)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        self.to(self.device)

    def forward(self, obs):
        features = self.state_processor(obs)
        return [action_head(features) for action_head in self.action_heads]

    def get_action(self, obs):
        logits = self(obs)
        actions = []
        for i in range(len(logits)):
            if self.action_discrete[i]:
                action_probs = nn.functional.log_softmax(logits[i], dim=0).exp()
                actions.append(torch.multinomial(action_probs, num_samples = 1).item())
            else:
                actions.append(torch.normal(logits[i][0], logits[i][1]).cpu().detach().numpy())
        '''
        for i in range(len(logits)):
            if self.action_discrete[i]:
                action_probs = nn.functional.log_softmax(logits[i], dim=1).exp()
                actions.append(torch.multinomial(action_probs, num_samples = 1)[:,0].cpu().detach().numpy())
            else:
                actions.append(torch.normal(logits[i][:,0], logits[i][:,1]).cpu().detach().numpy())
        '''
        return actions

    def get_log_prob(self, network_outputs, actions_taken):
        actions_taken = torch.Tensor(actions_taken).to(self.device)
        total_log_prob = 0
        for i in range(len(network_outputs)):
            if self.action_discrete[i]:
                action_probs = nn.functional.log_softmax(network_outputs[i], dim=1).exp()
                total_log_prob += torch.distributions.Categorical(action_probs).log_prob(actions_taken[:,i]).sum()
            else:
                total_log_prob += torch.distributions.Normal(network_outputs[i][:,0], network_outputs[i][:,1]).log_prob(actions_taken[:,i]).sum()
        return total_log_prob

    def update(self, obs, act, adv):
        network_outputs = self(torch.Tensor(obs).to(self.device))
        logprob_pi = self.get_log_prob(network_outputs, act)

        self.optimizer.zero_grad()
        loss = torch.sum((-logprob_pi * torch.Tensor(adv).to(self.device)))
        loss.backward()
        self.optimizer.step()

device = "cuda:0"
ITERATIONS = 1000000
discount = 0.99
agent = nn_agent(1024, device)
env = DerkEnv(n_arenas = 20, turbo_mode = True, reward_function = reward_function)

for i in range(ITERATIONS):
    print("-----------------------------ITERATION " + str(i) + "-----------------------------")
    observation = []
    done = []
    action = []
    reward = []

    observation_n = env.reset()
    while True:
        #get actions from the agents for all of the paralell games (and agents)
        action_n = [agent.get_action(torch.Tensor(observation_n[i]).to(device)) for i in range(env.n_agents)]
        #act in environment and observe the new obervation and reward (done tells you if episode is over)
        observation_n, reward_n, done_n, _ = env.step(action_n)

        #collect experience data to learn from
        observation.append(observation_n)
        reward.append(reward_n)
        done.append(done_n)
        action.append(action_n)

        if all(done_n):
          print("Episode finished")
          break

    # reshapes all collected data to [episode num, timestep]
    observation = np.swapaxes(np.array(observation), 0, 1)
    reward = np.swapaxes(np.array(reward), 0, 1)
    done = np.swapaxes(np.array(done), 0, 1)
    action = np.swapaxes(np.array(action), 0, 1)

    print("average reward: " + str(reward.mean()))

    #calculate discounted returns
    discounted_returns = np.zeros(reward.size)
    for i in range(reward.shape[0]):
        cumulative_reward = 0
        for j in range(reward.shape[1]):
            cumulative_reward += reward[i, reward.shape[1] - j - 1]
            discounted_returns[i*reward.shape[1] + reward.shape[1] - j - 1] = cumulative_reward
            cumulative_reward *= discount

    #normalize discounted returns
    norm_discounted_returns = (discounted_returns - discounted_returns.mean()) / discounted_returns.std()

    #reshape the observation and action data into one big batch
    observation = observation.reshape(-1, observation.shape[2])
    action = action.reshape(-1, action.shape[2])

    #learn from experience
    agent.update(observation, action, norm_discounted_returns)


env.close()
