from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np
import copy
import time
import os
from tqdm import tqdm

shaped_reward_function = {
    "damageEnemyStatue":  0.01,
    "damageEnemyUnit":  0.01,
    "killEnemyStatue": 0.1,
    "killEnemyUnit":  0.02,
    "healFriendlyStatue":  0.02,
    "healTeammate1": 0.01,
    "healTeammate2":  0.01,
    "timeSpentHomeBase": 0,
    "timeSpentHomeTerritory": 0,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 0,
    "damageTaken": - 0.01,
    "friendlyFire": - 0.01,
    "healEnemy": - 0.01,
    "fallDamageTaken": -.1,
    "statueDamageTaken": - 0.04,
    "manualBonus": 0,
    "victory": 2,
    "loss": -2,
    "tie": 0,
    "teamSpirit": 0.3,
    "timeScaling": 1.0,
}

win_loss_reward_function = {
    "damageEnemyStatue": 0,
    "damageEnemyUnit": 0,
    "killEnemyStatue": 0,
    "killEnemyUnit": 0,
    "healFriendlyStatue": 0.,
    "healTeammate1": 0,
    "healTeammate2": 0,
    "timeSpentHomeBase": 0,
    "timeSpentHomeTerritory": 0,
    "timeSpentAwayTerritory": 0,
    "timeSpentAwayBase": 0,
    "damageTaken": 0,
    "friendlyFire": 0,
    "healEnemy": 0,
    "fallDamageTaken": 0,
    "statueDamageTaken": 0,
    "manualBonus": 0,
    "victory": 0.3333333333333,
    "loss": 0,
    "tie": 0.16666666666666,
    "teamSpirit": 1.0,
    "timeScaling": 1.0,
}

classes_team_config = [
      { 'primaryColor': '#ff00ff', 'slots': ['Talons', 'FrogLegs', 'ParalyzingDart'] },
      { 'primaryColor': '#00ff00', 'slots': ['Magnum', 'Trombone', 'VampireGland'] },
      { 'primaryColor': '#ff0000', 'slots': ['Cripplers', 'IronBubblegum', 'HealingGland'] }
   ]


class nn_agent(nn.Module):
    def __init__(self, hidden_size, device, activation = nn.Tanh()):
        super().__init__()

        self.gamma = 0.99
        self.lamda = 0.99

        self.sgd_iterations = 1
        self.mini_batch_size = 1000
        self.recompute_every = 25 #how often to recompute hidden states and advantages. Expensive but allows more accurate training
        self.eps_clip = 0.1
        self.entropy_coeff = 0.01 #prevents policy collapse by keeping some randomness for exploration
        self.value_coeff = 0.1

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

        self.value_head = nn.Linear(hidden_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 5e-5)
        self.to(self.device)

    def forward(self, obs):
        features = self.state_processor(obs)
        continuous_actions = self.continuous_action_head(features)

        #bound action 0 and 1 between -1 and 1, bound action 2 between 0 and 1
        continuous_actions[:,0:2] = torch.tanh(continuous_actions[:,0:2])
        continuous_actions[:,2] = torch.sigmoid(continuous_actions[:,2])

        discrete_actions = [action_head(features) for action_head in self.discrete_action_heads]
        value = self.value_head(features)

        return continuous_actions, discrete_actions, value

    def get_action(self, obs):
        continuous_means, discrete_output, _ = self(obs)
        actions =  torch.normal(continuous_means, self.logstd.exp()).cpu().detach().numpy()

        for discrete_logits in discrete_output:
            discrete_probs = nn.functional.log_softmax(discrete_logits, dim=1).exp()
            discrete_actions = torch.multinomial(discrete_probs, num_samples = 1).flatten().cpu().detach().numpy()
            actions = np.concatenate((actions, discrete_actions.reshape(-1,1)), axis = 1)

        return actions

    def get_action_info(self, continuous_means, discrete_output, actions_taken):
        normal_dists = torch.distributions.Normal(continuous_means, self.logstd.exp())

        log_probs = normal_dists.log_prob(actions_taken[:,:self.continuous_size])
        entropy = normal_dists.entropy()

        for i in range(len(discrete_output)):
            discrete_probs = nn.functional.log_softmax(discrete_output[i], dim=1).exp()
            discrete_dist = torch.distributions.Categorical(discrete_probs)
            discrete_log_probs = discrete_dist.log_prob(actions_taken[:,self.continuous_size+i])

            log_probs = torch.cat((log_probs, discrete_log_probs.unsqueeze(1)), axis=1)
            entropy = torch.cat((entropy, discrete_dist.entropy().unsqueeze(1)), axis=1)

        return log_probs, entropy

    #calculates GAE given rew and values. Assumes it is getting complete episodes in shape [batch_size, timesteps]
    def calc_GAE(self, reward, value):
        value = value.reshape(reward.shape[0], reward.shape[1])
        value = np.concatenate((value, np.zeros(reward.shape[0]).reshape(-1, 1)), axis = 1)

        TD_errors = reward + self.gamma * value[:,1:] - value[:,:-1]
        #calculate discounted returns
        advantages = np.zeros(reward.size)
        for i in range(TD_errors.shape[0]):
            gae = 0
            for j in range(TD_errors.shape[1]):
                gae += TD_errors[i, TD_errors.shape[1] - j - 1]
                advantages[i*TD_errors.shape[1] + TD_errors.shape[1] - j - 1] = gae
                gae *= self.gamma * self.lamda

        return advantages

    def update(self, obs, act, rew):
        continuous_means, discrete_output, _ = self(torch.Tensor(obs).to(self.device))
        original_log_prob_pi, entropy = self.get_action_info(continuous_means, discrete_output, torch.Tensor(act).to(self.device))
        original_log_prob_pi = original_log_prob_pi.detach()

        print("Policy Entropy: ", entropy.mean(axis=0).detach().cpu().numpy().tolist())
        print("Policy Standev: ", self.logstd.exp().detach().cpu().numpy().tolist())
        print("\nTraining with PPO")

        for training_iteration in range(self.sgd_iterations):
            shuffled = torch.randperm(obs.shape[0])

            for minibatch_num in tqdm(range((obs.shape[0]//self.mini_batch_size) - 1)):
                #recompute advantages and hidden states periodically to keep updates accurate
                if minibatch_num % self.recompute_every == 0:
                    _, _, value = self(torch.Tensor(obs).to(self.device))
                    adv = self.calc_GAE(rew, value.detach().cpu().numpy())
                    value_targets = value.squeeze().detach().cpu().numpy() + adv
                    norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                shuffled_minibatch = shuffled[minibatch_num*self.mini_batch_size:(minibatch_num+1)*self.mini_batch_size]
                minibatch_act = torch.Tensor(act[shuffled_minibatch]).to(self.device)
                minibatch_norm_adv = torch.Tensor(norm_adv[shuffled_minibatch]).to(self.device)

                continuous_means, discrete_output, value = self(torch.Tensor(obs[shuffled_minibatch]).to(self.device))
                curr_logprob_pi, entropy = self.get_action_info(continuous_means, discrete_output, minibatch_act)
                ratio = torch.exp(curr_logprob_pi - original_log_prob_pi[shuffled_minibatch])

                value_loss = torch.pow(value - torch.Tensor(value_targets[shuffled_minibatch]).to(self.device), 2)

                surrogate_loss1 = ratio.sum(axis=1) * minibatch_norm_adv
                surrogate_loss2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip).sum(axis=1) * minibatch_norm_adv
                loss = -torch.min(surrogate_loss1, surrogate_loss2) - self.entropy_coeff * entropy.sum(axis=1) + self.value_coeff * value_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

device = "cuda:0"
ITERATIONS = 1000000
agent = nn_agent(512, device)
env = DerkEnv(n_arenas = 500, turbo_mode = True, reward_function = win_loss_reward_function, home_team = classes_team_config, away_team = classes_team_config)

past_selves_ratio = 0.2
save_model_every = 10
eval_against_gap = 100
past_models = []

portion_controlled_by_curr = 1 - (past_selves_ratio/2)

model_checkpoint_schedule = [int(i ** 1.5) for i in range(1000)]
save_folder = "checkpoints/PPO-GAE-" + str(time.time())
os.mkdir(save_folder)

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

    if iteration % save_model_every == 0:
        past_models.append(copy.deepcopy(agent))

    if iteration in model_checkpoint_schedule:
        torch.save(agent.state_dict(), save_folder + "/" + str(iteration))

    observation = []
    done = []
    action = []
    reward = []

    #picks past agents spaced further apart based on how far into training you are
    past_selve_indices = [(int(iteration - (i * (iteration ** 0.7))) // save_model_every) for i in range(4)]

    observation_n = env.reset()
    while True:
        action_n = agent.get_action(torch.Tensor(observation_n[:int(env.n_agents*portion_controlled_by_curr)]).to(device))

        #quick and dirty way to query actions from past agents in a 0.5:0.25:0.25 ratio
        #should be changed to be more general/ less terrible later
        past_agent_1_action = past_models[past_selve_indices[0]].get_action(torch.Tensor \
        (observation_n[int(env.n_agents*portion_controlled_by_curr):int(env.n_agents*(portion_controlled_by_curr+past_selves_ratio*0.5))]).to(device))
        past_agent_2_action = past_models[past_selve_indices[1]].get_action(torch.Tensor \
        (observation_n[int(env.n_agents*(portion_controlled_by_curr+past_selves_ratio*0.5)):int(env.n_agents*(portion_controlled_by_curr+past_selves_ratio*0.75))]).to(device))
        past_agent_3_action = past_models[past_selve_indices[2]].get_action(torch.Tensor \
        (observation_n[int(env.n_agents*(portion_controlled_by_curr+past_selves_ratio*0.75)):]).to(device))
        action_n = np.concatenate((action_n, past_agent_1_action, past_agent_2_action, past_agent_3_action), axis = 0)

        #act in environment and observe the new obervation and reward (done tells you if episode is over)
        observation_n, reward_n, done_n, _ = env.step(action_n)

        #collect experience data to learn from
        observation.append(observation_n[:int(env.n_agents*portion_controlled_by_curr)])
        reward.append(reward_n[:int(env.n_agents*portion_controlled_by_curr)])
        done.append(done_n[:int(env.n_agents*portion_controlled_by_curr)])
        action.append(action_n[:int(env.n_agents*portion_controlled_by_curr)])

        if all(done_n):
          break

    # reshapes all collected data to [episode num, timestep]
    observation = np.swapaxes(np.array(observation), 0, 1)
    reward = np.swapaxes(np.array(reward), 0, 1)
    done = np.swapaxes(np.array(done), 0, 1)
    action = np.swapaxes(np.array(action), 0, 1)

    #reshape the observation and action data into one big batch
    observation = observation.reshape(-1, observation.shape[2])
    action = action.reshape(-1, action.shape[2])

    #learn from experience
    print("Training with PPO")
    agent.update(observation, action, reward)

    if iteration % 2 == 0:
        testing_against = max(0, (iteration - eval_against_gap) // save_model_every)
        if iteration % 4 == 0:
            testing_against = 0
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
