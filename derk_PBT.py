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
from PBT import *

torch.autograd.set_detect_anomaly(True)

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

class lstm_agent(PBTAgent):
    def __init__(self, lstm_size, device, activation = nn.Tanh(), hyperparams = {}):
        super().__init__()
        self.device = device

        #### HYPERPARAMETERS ####
        default_hyperparams = {
            "learning_rate": 2e-3,
            # discount factor, measure of how much you care about rewards in the future vs now
            # should probably be 1.0 for pure win-loss rewards
            "gamma": 1.0,
            # whether or not to use Generalized Advantage Estimation (GAE). Allows a flexible tradeoff
            # between bias (value predictions) and variance (true returns), but requires
            # training the value network
            "use_gae": True,
            # lambda param for GAE estimation, defines the tradeoff between bias
            # (using the value function) and variance (using actual returns)
            "lambda": 0.96,
            # how much to optimize the value function, too much interferes with policy
            # leaning, too little and value function won't be accurate
            "value_coeff": 0.5,
            # fragment size determines how many sequential steps we chunk experience into
            # larger size allows more flow of gradients back in time but makes batches less diverse
            # must be a factor of 150 or else some experience will be cut off
            "lstm_fragment_length": 15,
            #how many times to loop over the entire batch of experience
            "epochs_per_update": 2,
            # how often to recompute hidden states and advantages. Expensive but allows more accurate training
            "recompute_every": 20,
            # defines size of trust region, smaller generally means more stable but slower learning
            "eps_clip": 0.2,
            # how much to optimize for entropy, prevents policy collapse by keeping some randomness for exploration
            "entropy_coeff": 0.01,
        }

        self.hyperparams = hyperparams.copy()
        for hyperparam_key in default_hyperparams.keys():
            if hyperparam_key not in hyperparams:
                self.hyperparams[hyperparam_key] = default_hyperparams[hyperparam_key]

        if not self.hyperparams["use_gae"]:
            #if not using value network, don't train it and set lambda decay = 1.0
            self.hyperparams["lambda"] = 1.0
            self.hyperparams["value_coeff"] = 0.0

        #batch size in fragments
        self.fragments_per_batch = 900 // self.hyperparams["lstm_fragment_length"]

        # whether to treat each component of an action as independent or not.
        # by default this should be set to False
        self.independent_action_components = False

        #### ARCHITECTURE ####

        self.fc = nn.Sequential(nn.Linear(64, lstm_size), activation)
        self.lstm = nn.LSTM(lstm_size, lstm_size, batch_first = True)

        self.continuous_size = 3
        self.logstd = nn.Parameter(torch.Tensor([0.693147, 0.693147, 0]))
        self.continuous_lower_bounds = torch.Tensor([-1, -1, 0]).to(self.device)
        self.continuous_upper_bounds = torch.Tensor([1, 1, 1]).to(self.device)
        self.continuous_action_head = nn.Sequential(activation, nn.Linear(lstm_size, self.continuous_size))

        self.discrete_action_heads = nn.ModuleList([nn.Sequential(activation, nn.Linear(lstm_size, 4)),
                                                    nn.Sequential(activation, nn.Linear(lstm_size, 8))])

        self.value_head = nn.Linear(lstm_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.hyperparams["learning_rate"])

        self.to(self.device)

    def get_hyperparams(self):
        return self.hyperparams

    def update_hyperparams(self, hyperparams_changed, evoke_update_event):
        if evoke_update_event:
            # Print the change in hyperparameters.
            print("Agent hyperparameters changed. Original:")
            hyperparams_changing = {}
            for changed_key in hyperparams_changed.keys():
                hyperparams_changing[changed_key] = self.hyperparams[changed_key]
            print(hyperparams_changing)

            print("New:")
            print(hyperparams_changed)

        for changed_key in hyperparams_changed.keys():
            self.hyperparams[changed_key] = hyperparams_changed[changed_key]
            
        if evoke_update_event:
            self.on_hyperparam_change(hyperparams_changed.keys())

    def on_hyperparam_change(self, hyperparams_changed):
        """To be run every time the hyperparameters of the model change.

        Args:
            hyperparams_changed (List[str]): A list of the keys of the hyperparameters changed.
        """
        if "learning_rate" in hyperparams_changed:
            # TODO: Consider the implications of resetting the internal optimizer state.
            self.optimizer = torch.optim.Adam(self.parameters(), lr = self.hyperparams["learning_rate"])

    def forward(self, obs, state):
        if len(obs.shape) < 3:
            lstm_in = self.fc(torch.Tensor(obs).to(self.device)).unsqueeze(1)
        else:
            lstm_in = self.fc(torch.Tensor(obs).to(self.device))

        if state == None:
            lstm_out, state = self.lstm(lstm_in)
        else:
            if len(state[0].shape) < 3:
                state[0] = state[0].unsqueeze(0)
                state[1] = state[1].unsqueeze(0)
            lstm_out, state = self.lstm(lstm_in, state)

        features = lstm_out.reshape(-1, lstm_out.shape[2]) #flattens batch and seq len dimension together

        continuous_vals = self.continuous_action_head(features)
        #bound action 0 and 1 between -1 and 1, bound action 2 between 0 and 1
        continuous_vals[:,0:2] = torch.tanh(continuous_vals[:,0:2])
        continuous_vals[:,2] = torch.sigmoid(continuous_vals[:,2])

        discrete_actions = [action_head(features) for action_head in self.discrete_action_heads]
        value = self.value_head(features)

        return continuous_vals, discrete_actions, value, state

    def get_action(self, obs, state):
        continuous_means, discrete_output, _, state = self(obs, state)

        truncNorm = TruncatedNormal(self.device, continuous_means, self.logstd.exp(), self.continuous_lower_bounds, self.continuous_upper_bounds)
        actions = truncNorm.rsample().detach().cpu().numpy()

        for discrete_logits in discrete_output:
            discrete_probs = nn.functional.log_softmax(discrete_logits, dim=1).exp()
            discrete_actions = torch.multinomial(discrete_probs, num_samples = 1).flatten().cpu().detach().numpy()
            actions = np.concatenate((actions, discrete_actions.reshape(-1,1)), axis = 1)

        return actions, state

    def get_action_info(self, continuous_means, discrete_output, actions_taken):
        truncNorm = TruncatedNormal(self.device, continuous_means, self.logstd.exp(), self.continuous_lower_bounds, self.continuous_upper_bounds)
        log_probs = truncNorm.log_prob(actions_taken[:,:self.continuous_size])
        entropy = truncNorm._entropy

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

        TD_errors = reward + self.hyperparams['gamma'] * value[:,1:] - value[:,:-1]

        advantages = np.zeros(reward.size)
        for i in range(TD_errors.shape[0]):
            gae = 0
            for j in range(TD_errors.shape[1]):
                gae += TD_errors[i, TD_errors.shape[1] - j - 1]
                advantages[i*TD_errors.shape[1] + TD_errors.shape[1] - j - 1] = gae
                gae *= self.hyperparams['gamma'] * self.hyperparams['lambda']

        return advantages

    def update(self, obs, act, rew):
        act = act.reshape(-1, act.shape[2]) #flatten actions

        continuous_means, discrete_output, _, _ = self(obs, None)
        original_log_prob_pi, entropy = self.get_action_info(continuous_means, discrete_output, torch.Tensor(act).to(self.device))
        original_log_prob_pi = original_log_prob_pi.detach()

        print("Policy Entropy: ", entropy.mean(axis=0).detach().cpu().numpy().tolist())
        print("Policy Standev: ", self.logstd.exp().detach().cpu().numpy().tolist())

        #break observation into fragments
        obs_fragmented = obs.reshape((obs.shape[0] * obs.shape[1]) // self.hyperparams["lstm_fragment_length"], self.hyperparams["lstm_fragment_length"], 64)

        for epoch in range(self.hyperparams['epochs_per_update']):
            shuffled = torch.randperm(obs_fragmented.shape[0])

            print("\nTraining PPO epoch", epoch)
            for minibatch_num in tqdm(range((obs_fragmented.shape[0]//self.fragments_per_batch) - 1)):
                #recompute advantages and hidden states periodically to keep updates accurate
                if minibatch_num % self.hyperparams['recompute_every'] == 0:
                    _, _, frag_value, frag_state = self(obs[:, :self.hyperparams["lstm_fragment_length"], :], None)

                    #initial zeroed out states
                    state = [frag_state[0].detach().cpu() * 0, frag_state[1].detach().cpu() * 0]

                    value = frag_value.reshape(-1, self.hyperparams["lstm_fragment_length"]).unsqueeze(1)

                    for i in range((150 // self.hyperparams["lstm_fragment_length"]) - 1):
                        #concatenate next fragment batch of states
                        #written before getting new one because we don't care about
                        #the last hidden state and put in an extra 0s state above
                        state[0] = torch.cat((state[0], frag_state[0].detach().cpu()), axis = 0)
                        state[1] = torch.cat((state[1], frag_state[1].detach().cpu()), axis = 0)

                        #get value and hidden states for next batch of fragments
                        _, _, frag_value, frag_state = self(obs[:, self.hyperparams["lstm_fragment_length"]*(i+1):self.hyperparams["lstm_fragment_length"]*(i+2), :], frag_state)

                        #concatenate next fragment batch of values
                        value = torch.cat((value, frag_value.reshape(-1, self.hyperparams["lstm_fragment_length"]).unsqueeze(1)), axis = 1)

                    #reshape so that the states correspond to a flattened [episode num, timestep]
                    state[0] = torch.flatten(state[0].permute(1, 0, 2), end_dim = 1)
                    state[1] = torch.flatten(state[1].permute(1, 0, 2), end_dim = 1)
                    value = torch.flatten(value, start_dim = 1)

                    if self.hyperparams["use_gae"] == False:
                        # calcualting GAE with lambda = 1.0 and value = 0 is the
                        # same as calcualting true returns
                        value *= 0

                    #make new calculations of advantage and value targets
                    adv = self.calc_GAE(rew, value.detach().cpu().numpy())
                    value_targets = torch.flatten(value).squeeze().detach().cpu().numpy() + adv
                    norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                #subset of shuffled indices to use in this minibatch
                shuffled_fragments = shuffled[minibatch_num*self.fragments_per_batch:(minibatch_num+1)*self.fragments_per_batch]
                shuffled_indices = torch.cat([((shuffled_fragments * self.hyperparams["lstm_fragment_length"]) + i) for i in range(self.hyperparams["lstm_fragment_length"])], axis = 0)

                #get actions and normalized advantages for this minibatch
                minibatch_act = torch.Tensor(act[shuffled_indices]).to(self.device)
                minibatch_norm_adv = torch.Tensor(norm_adv[shuffled_indices]).to(self.device)

                #initial states for this minibatch
                minibatch_state = [state[0][shuffled_fragments].to(self.device), state[1][shuffled_fragments].to(self.device)]
                continuous_means, discrete_output, value, _ = self(obs_fragmented[shuffled_fragments], minibatch_state)
                curr_logprob_pi, entropy = self.get_action_info(continuous_means, discrete_output, minibatch_act)

                #PPO clipped policy loss
                if self.independent_action_components:
                    log_ratio = curr_logprob_pi - original_log_prob_pi[shuffled_indices]

                    #clip out massive ratios so they don't produce inf values
                    clipped_log_ratio = torch.clamp(log_ratio, -80, 80)
                    ratio = torch.exp(clipped_log_ratio)

                    surrogate_loss1 = ratio * minibatch_norm_adv.unsqueeze(1)
                    surrogate_loss2 = torch.clamp(ratio, 1 - self.hyperparams['eps_clip'], 1 + self.hyperparams['eps_clip']) * minibatch_norm_adv.unsqueeze(1)
                else:
                    log_ratio = curr_logprob_pi.sum(axis=1) - original_log_prob_pi[shuffled_indices].sum(axis=1)

                    #clip out massive ratios so they don't produce inf values
                    clipped_log_ratio = torch.clamp(log_ratio, -80, 80)
                    ratio = torch.exp(clipped_log_ratio)

                    surrogate_loss1 = ratio * minibatch_norm_adv
                    surrogate_loss2 = torch.clamp(ratio, 1 - self.hyperparams['eps_clip'], 1 + self.hyperparams['eps_clip']) * minibatch_norm_adv

                #loss for the value function
                value_loss = torch.pow(value - torch.Tensor(value_targets[shuffled_indices]).to(self.device), 2)

                loss = -torch.min(surrogate_loss1, surrogate_loss2).mean() - self.hyperparams['entropy_coeff'] * entropy.mean() + self.hyperparams['value_coeff'] * value_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

device = "cuda:0"
ITERATIONS = 1000000
agent = lstm_agent(512, device)

arm_weapons = ["Talons", "BloodClaws", "Cleavers", "Cripplers", "Pistol", "Magnum", "Blaster"]
misc_weapons = ["FrogLegs", "IronBubblegum", "HeliumBubblegum", "Shell", "Trombone"]
tail_weapons = ["HealingGland", "VampireGland", "ParalyzingDart"]

n_arenas = 80
random_configs = [{"slots": [random.choice(arm_weapons), random.choice(misc_weapons), random.choice(tail_weapons)]} for i in range(3 * n_arenas // 2)]
env = DerkEnv(n_arenas = n_arenas, turbo_mode = True, reward_function = win_loss_reward_function, home_team = random_configs, away_team = random_configs)

# PBT Parameters
population_size = 10
pbt_min_iterations = 10
# Define which hyperparameters to exploit and how to copy the values.
exploit_methods = {
    'learning_rate': lambda x: x,
    'entropy_coeff': lambda x: x,
    'value_coeff': lambda x: x,
}
# Define which hyperparameters to explore and how to.
perturb_explore = get_perturb_explore()
explore_methods = {
    'learning_rate': perturb_explore,
    'entropy_coeff': perturb_explore,
    'value_coeff': perturb_explore,
}

teams_per_member = (env.n_agents // 3) // population_size
# Initialize population with uniformly distributed hyperparameters.
population = [lstm_agent(512, device, hyperparams={
    'learning_rate': np.random.uniform(0.0001, 0.01),
    'entropy_coeff': np.random.uniform(0.001, 0.1),
    'value_coeff': np.random.uniform(0.25, 1)
}) for i in range(population_size)]
# Record the last PBT update
last_PBT_update = [0] * len(population)

model_checkpoint_schedule = [2*int(i ** 1.6) for i in range(1000)]
save_folder = "checkpoints/PPO-LSTM-PBT" + str(time.time())
os.makedirs(save_folder)

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

    observation = [[] for i in range(population_size)]
    action = [[] for i in range(population_size)]
    reward = [[] for i in range(population_size)]
    states = [None for i in range(population_size)]

    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5))
            for i in range(population_size):
                action_n[league_agent_mappings[i]], states[i] = population[i].get_action(observation_n[league_agent_mappings[i]], states[i])

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

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

env.close()
