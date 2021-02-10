from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np
import copy

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
    def __init__(self, hidden_size, device, activation = nn.ReLU()):
        super().__init__()
        self.device = device

        self.state_processor = nn.Sequential(nn.Linear(64, hidden_size),
                                             activation,
                                             nn.Linear(hidden_size, hidden_size),
                                             activation)

        self.continuous_size = 3
        self.logstd = nn.Parameter(torch.zeros(self.continuous_size))
        self.continuous_action_head = nn.Linear(hidden_size, self.continuous_size)

        self.discrete_action_heads = nn.ModuleList([nn.Linear(hidden_size, 4),
                                                    nn.Linear(hidden_size, 8)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 5e-5)
        self.to(self.device)

    def forward(self, obs):
        features = self.state_processor(obs)
        continuous_actions = self.continuous_action_head(features)

        #bound action 0 and 1 between -1 and 1, bound action 2 between 0 and 1
        continuous_actions[:,0:2] = torch.tanh(continuous_actions[:,0:2])
        continuous_actions[:,2] = torch.sigmoid(continuous_actions[:,2])

        discrete_actions = [action_head(features) for action_head in self.discrete_action_heads]
        return continuous_actions, discrete_actions

    def get_action(self, obs):
        continuous_means, discrete_output = self(obs)
        actions =  torch.normal(continuous_means, self.logstd.exp()).cpu().detach().numpy()

        for discrete_logits in discrete_output:
            discrete_probs = nn.functional.log_softmax(discrete_logits, dim=1).exp()
            discrete_actions = torch.multinomial(discrete_probs, num_samples = 1).flatten().cpu().detach().numpy()
            actions = np.concatenate((actions, discrete_actions.reshape(-1,1)), axis = 1)

        return actions

    def get_log_prob(self, obs, actions_taken):
        continuous_means, discrete_output = self(torch.Tensor(obs).to(self.device))

        total_log_prob = torch.distributions.Normal(continuous_means, self.logstd.exp()).log_prob(actions_taken[:,0:self.continuous_size]).sum(axis=1)

        for i in range(len(discrete_output)):
            discrete_probs = nn.functional.log_softmax(discrete_output[i], dim=1).exp()
            total_log_prob += torch.distributions.Categorical(discrete_probs).log_prob(actions_taken[:,self.continuous_size+i])

        return total_log_prob

    def update(self, obs, act, adv):
        logprob_pi = self.get_log_prob(obs, torch.Tensor(act).to(self.device))

        self.optimizer.zero_grad()
        loss = torch.sum((-logprob_pi * torch.Tensor(adv).to(self.device)))
        loss.backward()
        self.optimizer.step()

device = "cuda:0"
ITERATIONS = 1000000
discount = 0.99
agent = nn_agent(512, device)
env = DerkEnv(n_arenas = 400, turbo_mode = True, reward_function = reward_function)

save_model_every = 10
play_against_gap = 30
past_models = []

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

    if iteration % save_model_every == 0:
        past_models.append(copy.deepcopy(agent))

    observation = []
    done = []
    action = []
    reward = []

    observation_n = env.reset()
    while True:
        action_n = agent.get_action(torch.Tensor(observation_n).to(device))
        #act in environment and observe the new obervation and reward (done tells you if episode is over)
        observation_n, reward_n, done_n, _ = env.step(action_n)

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

    #calculate discounted returns
    discounted_returns = np.zeros(reward.size)
    for i in range(reward.shape[0]):
        cumulative_reward = 0
        for j in range(reward.shape[1]):
            cumulative_reward += reward[i, reward.shape[1] - j - 1]
            discounted_returns[i*reward.shape[1] + reward.shape[1] - j - 1] = cumulative_reward
            cumulative_reward *= discount

    #normalize discounted returns
    norm_discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

    #reshape the observation and action data into one big batch
    observation = observation.reshape(-1, observation.shape[2])
    action = action.reshape(-1, action.shape[2])

    #learn from experience
    agent.update(observation, action, norm_discounted_returns)

    if iteration % 2 == 0:
        testing_against = max(0, (iteration - play_against_gap) // save_model_every)
        print("\nEvaluating against iteration ", testing_against * save_model_every)

        observation_n = env.reset()
        while True:
            #get actions for agent (first half of observations) and random agent (second half of observations)
            agent_action_n = agent.get_action(torch.Tensor(observation_n[:env.n_agents//2]).to(device)).tolist()
            random_action_n = past_models[testing_against].get_action(torch.Tensor(observation_n[env.n_agents//2:]).to(device)).tolist()
            action_n = agent_action_n + random_action_n

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

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

env.close()
