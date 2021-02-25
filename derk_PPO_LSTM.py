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
      { 'primaryColor': '#ff0000', 'slots': ['Cripplers', 'IronBubblegum', 'HealingGland'] }]

class lstm_agent(nn.Module):
    def __init__(self, lstm_size, device, activation = nn.Tanh()):
        super().__init__()
        self.device = device

        #### HYPERPARAMETERS ####

        self.learning_rate = 5e-3
        # discount factor, measure of how much you care about rewards in the future vs now
        # should probably be 1.0 for pure win-loss rewards
        self.gamma = 1.0
        # whether or not to use Generalized Advantage Estimation (GAE). Allows a flexible tradeoff
        # between bias (value predictions) and variance (true returns), but requires
        # training the value network
        self.use_gae = False
        # we only train value network and use lambda param if we are using GAE
        if self.use_gae:
            # lambda param for GAE estimation, defines the tradeoff between bias
            # (using the value function) and variance (using actual returns)
            self.lamda = 0.99
            # how much to optimize the value function, too much interferes with policy
            # leaning, too little and value function won't be accurate
            self.value_coeff = 0.5
        else:
            #if not using value network, don't train it and set lambda decay = 1.0
            self.lamda = 1.0
            self.value_coeff = 0.0
        # fragment size determines how many sequential steps we chunk experience into
        # larger size allows more flow of gradients back in time but makes batches less diverse
        # must be a factor of 150 or else some experience will be cut off
        self.lstm_fragment_length = 10
        #batch size in fragments
        self.fragments_per_batch = 1000 // self.lstm_fragment_length
        #how many times to loop over the entire batch of experience
        self.epochs_per_update = 1
        # how often to recompute hidden states and advantages. Expensive but allows more accurate training
        self.recompute_every = 20
        # defines size of trust region, smaller generally means more stable but slower learning
        self.eps_clip = 0.2
        # how much to optimize for entropy, prevents policy collapse by keeping some randomness for exploration
        self.entropy_coeff = 0.001

        #### ARCHITECTURE ####

        self.lstm = nn.LSTM(64, lstm_size, batch_first = True)

        self.continuous_size = 3
        self.logstd = nn.Parameter(torch.zeros(self.continuous_size))
        self.continuous_action_head = nn.Sequential(activation, nn.Linear(lstm_size, self.continuous_size))

        self.discrete_action_heads = nn.ModuleList([nn.Sequential(activation, nn.Linear(lstm_size, 4)),
                                                    nn.Sequential(activation, nn.Linear(lstm_size, 8))])

        self.value_head = nn.Linear(lstm_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        self.to(self.device)

    def forward(self, obs, state):
        if len(obs.shape) < 3:
            lstm_in = torch.Tensor(obs).to(self.device).unsqueeze(1)
        else:
            lstm_in = torch.Tensor(obs).to(self.device)

        if state == None:
            lstm_out, state = self.lstm(lstm_in)
        else:
            if len(state[0].shape) < 3:
                state[0] = state[0].unsqueeze(0)
                state[1] = state[1].unsqueeze(0)
            lstm_out, state = self.lstm(lstm_in, state)

        features = lstm_out.reshape(-1, lstm_out.shape[2]) #flattens batch and seq len dimension together

        continuous_actions = self.continuous_action_head(features)
        #bound action 0 and 1 between -1 and 1, bound action 2 between 0 and 1
        continuous_actions[:,0:2] = torch.tanh(continuous_actions[:,0:2])
        continuous_actions[:,2] = torch.sigmoid(continuous_actions[:,2])

        discrete_actions = [action_head(features) for action_head in self.discrete_action_heads]
        value = self.value_head(features)

        return continuous_actions, discrete_actions, value, state

    def get_action(self, obs, state):
        continuous_means, discrete_output, _, state = self(obs, state)
        actions =  torch.normal(continuous_means, torch.clip(self.logstd.exp(), 0.002, 2)).cpu().detach().numpy()

        for discrete_logits in discrete_output:
            discrete_probs = nn.functional.log_softmax(discrete_logits, dim=1).exp()
            discrete_actions = torch.multinomial(discrete_probs, num_samples = 1).flatten().cpu().detach().numpy()
            actions = np.concatenate((actions, discrete_actions.reshape(-1,1)), axis = 1)

        return actions, state

    def get_action_info(self, continuous_means, discrete_output, actions_taken):
        normal_dists = torch.distributions.Normal(continuous_means, torch.clip(self.logstd.exp(), 0.002, 2))

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
        value = np.concatenate((value, np.zeros(reward.shape[0]).reshape(-1, 1)), axis = 1)

        TD_errors = reward + self.gamma * value[:,1:] - value[:,:-1]

        advantages = np.zeros(reward.size)
        for i in range(TD_errors.shape[0]):
            gae = 0
            for j in range(TD_errors.shape[1]):
                gae += TD_errors[i, TD_errors.shape[1] - j - 1]
                advantages[i*TD_errors.shape[1] + TD_errors.shape[1] - j - 1] = gae
                gae *= self.gamma * self.lamda

        return advantages

    def update(self, obs, act, rew):
        continuous_means, discrete_output, _, _ = self(obs, None)
        original_log_prob_pi, entropy = self.get_action_info(continuous_means, discrete_output, torch.Tensor(act.reshape(-1, act.shape[2])).to(self.device))
        original_log_prob_pi = original_log_prob_pi.detach()

        print("Policy Entropy: ", entropy.mean(axis=0).detach().cpu().numpy().tolist())
        print("Policy Standev: ", torch.clip(self.logstd.exp(), 0.002, 2).detach().cpu().numpy().tolist())

        #break observation into fragments
        obs_fragmented = obs.reshape((obs.shape[0] * obs.shape[1]) // self.lstm_fragment_length, self.lstm_fragment_length, 64)
        act = act.reshape(-1, act.shape[2])

        for epoch in range(self.epochs_per_update):
            shuffled = torch.randperm(obs_fragmented.shape[0])

            print("\nTraining PPO epoch", epoch)
            for minibatch_num in tqdm(range((obs_fragmented.shape[0]//self.fragments_per_batch) - 1)):
                #recompute advantages and hidden states periodically to keep updates accurate
                if minibatch_num % self.recompute_every == 0:
                    _, _, frag_value, frag_state = self(obs[:, :self.lstm_fragment_length, :], None)

                    #initial zeroed out states
                    state = [frag_state[0].detach().cpu() * 0, frag_state[1].detach().cpu() * 0]

                    value = frag_value.reshape(-1, self.lstm_fragment_length).unsqueeze(1)

                    for i in range((150 // self.lstm_fragment_length) - 1):
                        #concatenate next fragment batch of states
                        #written before getting new one because we don't care about
                        #the last hidden state and put in an extra 0s state above
                        state[0] = torch.cat((state[0], frag_state[0].detach().cpu()), axis = 0)
                        state[1] = torch.cat((state[1], frag_state[1].detach().cpu()), axis = 0)

                        #get value and hidden states for next batch of fragments
                        _, _, frag_value, frag_state = self(obs[:, self.lstm_fragment_length*(i+1):self.lstm_fragment_length*(i+2), :], frag_state)

                        #concatenate next fragment batch of values
                        value = torch.cat((value, frag_value.reshape(-1, self.lstm_fragment_length).unsqueeze(1)), axis = 1)

                    #reshape so that the states correspond to a flattened [episode num, timestep]
                    state[0] = torch.flatten(state[0].permute(1, 0, 2), end_dim = 1)
                    state[1] = torch.flatten(state[1].permute(1, 0, 2), end_dim = 1)
                    value = torch.flatten(value, start_dim = 1)

                    if self.use_gae == False:
                        # calcualting GAE with lambda = 1.0 and value = 0 is the
                        # same as calcualting true returns
                        value *= 0

                    #make new calculations of advantage and value targets
                    adv = self.calc_GAE(rew, value.detach().cpu().numpy())
                    value_targets = torch.flatten(value).squeeze().detach().cpu().numpy() + adv
                    norm_adv = (adv - adv.mean()) / adv.std()

                #subset of shuffled indices to use in this minibatch
                shuffled_fragments = shuffled[minibatch_num*self.fragments_per_batch:(minibatch_num+1)*self.fragments_per_batch]
                shuffled_indices = torch.cat([((shuffled_fragments * self.lstm_fragment_length) + i) for i in range(self.lstm_fragment_length)], axis = 0)

                #get actions and normalized advantages for this minibatch
                minibatch_act = torch.Tensor(act[shuffled_indices]).to(self.device)
                minibatch_norm_adv = torch.Tensor(norm_adv[shuffled_indices]).to(self.device)

                #initial states for this minibatch
                minibatch_state = [state[0][shuffled_fragments].to(self.device), state[1][shuffled_fragments].to(self.device)]
                continuous_means, discrete_output, value, _ = self(obs_fragmented[shuffled_fragments], minibatch_state)
                curr_logprob_pi, entropy = self.get_action_info(continuous_means, discrete_output, minibatch_act)
                ratio = torch.exp(curr_logprob_pi - original_log_prob_pi[shuffled_indices])

                #loss for the value function
                value_loss = torch.pow(value - torch.Tensor(value_targets[shuffled_indices]).to(self.device), 2)

                #PPO clipped policy loss
                surrogate_loss1 = ratio.sum(axis = 1) * minibatch_norm_adv
                surrogate_loss2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip).sum(axis=1) * minibatch_norm_adv
                loss = -torch.min(surrogate_loss1, surrogate_loss2) - self.entropy_coeff * entropy.sum(axis=1) + self.value_coeff * value_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

device = "cuda:0"
ITERATIONS = 1000000
agent = lstm_agent(512, device)
env = DerkEnv(n_arenas = 100, turbo_mode = True, reward_function = win_loss_reward_function, home_team = classes_team_config, away_team = classes_team_config)

save_model_every = 100
eval_against_gap = 100
past_models = []

#past_selves_ratio = 0.2
#portion_controlled_by_curr = 1 - (past_selves_ratio/2)

model_checkpoint_schedule = [int(i ** 1.5) for i in range(1000)]
save_folder = "checkpoints/PPO-LSTM-" + str(time.time())
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
    state = None

    #picks past agents spaced further apart based on how far into training you are
    past_selve_indices = [(int(iteration - (i * (iteration ** 0.7))) // save_model_every) for i in range(4)]

    observation_n = env.reset()
    while True:
        action_n, state = agent.get_action(observation_n, state)

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

        observation_n = env.reset()
        while True:
            #get actions for agent (first half of observations) and random agent (second half of observations)
            agent_action_n, curr_agent_state = agent.get_action(observation_n[:env.n_agents//2], curr_agent_state)
            past_action_n, past_agent_state = past_models[testing_against].get_action(observation_n[env.n_agents//2:], past_agent_state)
            action_n = agent_action_n.tolist() + past_action_n.tolist()

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
