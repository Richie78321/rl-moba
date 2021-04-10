from gym_derk.envs import DerkEnv
import time
import torch
from torch import nn
import numpy as np
import copy
import time
import os
import random
from math import ceil
from tqdm import tqdm
from torch_truncnorm.TruncatedNormal import TruncatedNormal
from PBT import *

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
            "lambda": 0.98,
            # how much to optimize the value function, too much interferes with policy
            # leaning, too little and value function won't be accurate
            "value_coeff": 0.5,
            # fragment size determines how many sequential steps we chunk experience into
            # larger size allows more flow of gradients back in time but makes batches less diverse
            # must be a factor of 150 or else some experience will be cut off
            "lstm_fragment_length": 15,
            #minibatch size (will be rounded up to nearest multiple of lstm_fragment_length)
            "minibatch_size": 900,
            #how many times to loop over the entire batch of experience
            "epochs_per_update": 2,
            # how often to recompute hidden states and advantages. Expensive but allows more accurate training
            "recompute_every": 10000,
            # defines size of trust region, smaller generally means more stable but slower learning
            "eps_clip": 0.2,
            # how much to optimize for entropy, prevents policy collapse by keeping some randomness for exploration
            "discrete_entropy_coeff": 1e-4,
            "continuous_entropy_coeff": 1e-4,
        }

        self.hyperparams = hyperparams.copy()
        for hyperparam_key in default_hyperparams.keys():
            if hyperparam_key not in hyperparams:
                self.hyperparams[hyperparam_key] = default_hyperparams[hyperparam_key]

        if not self.hyperparams["use_gae"]:
            #if not using value network, don't train it and set lambda decay = 1.0
            self.hyperparams["lambda"] = 1.0
            self.hyperparams["value_coeff"] = 0.0

        # whether to clip each component of an action as independently or not.
        # by default this should be set to False
        self.clip_independently = False

        # The move and rotate actions are only executed in proportion to the chase
        # action, so we multiply the gradients of move and rotate by this proportion
        # by default this should be set to True
        self.dependent_movement_actions = True

        # Combines the discrete heads of size 4 and 8 into one head of 32
        # (Consisting of every combination of focus/weapon)
        self.combine_discrete_moves = False

        #### ARCHITECTURE ####

        self.fc = nn.Sequential(nn.Linear(64, lstm_size), activation)
        self.lstm = nn.LSTM(lstm_size, lstm_size, batch_first = True)

        self.value_head = nn.Linear(lstm_size, 1)

        self.continuous_size = 3
        self.continuous_lower_bounds = torch.Tensor([-1, -1, 0]).to(self.device)
        self.continuous_upper_bounds = torch.Tensor([1, 1, 1]).to(self.device)

        if self.combine_discrete_moves == True:
            self.continuous_mean_head = nn.Sequential(activation, nn.Linear(lstm_size + 32, self.continuous_size))
            self.continuous_std_head = nn.Sequential(activation, nn.Linear(lstm_size + 32, self.continuous_size))
            self.discrete_action_head = nn.Sequential(activation, nn.Linear(lstm_size, 32))

        else:
            self.continuous_mean_head = nn.Sequential(activation, nn.Linear(lstm_size + 8, self.continuous_size))
            self.continuous_std_head = nn.Sequential(activation, nn.Linear(lstm_size + 8, self.continuous_size))
            self.focus_action_head = nn.Sequential(activation, nn.Linear(lstm_size, 8))
            #All other heads will take in one hot encoding of the focus action
            self.ability_action_head = nn.Sequential(activation, nn.Linear(lstm_size + 8, 4))

        #self.optimizer = torch.optim.Adam(self.parameters(), lr = self.hyperparams["learning_rate"])
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr = self.hyperparams["learning_rate"])

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
            # https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
            for g in self.optimizer.param_groups:
                g['lr'] = self.hyperparams["learning_rate"]

    def forward(self, obs, state, actions_taken = None):
        # If obs is not a tensor make a tensor and move to device
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)
        # If obs does not include a time dim add it
        if len(obs.shape) < 3:
            lstm_in = self.fc(obs).unsqueeze(1)
        else:
            lstm_in = self.fc(obs)
        # Use state if it was given otherwise use default all 0s state
        if state == None:
            lstm_out, state = self.lstm(lstm_in)
        else:
            if len(state[0].shape) < 3:
                state[0] = state[0].unsqueeze(0)
                state[1] = state[1].unsqueeze(0)
            lstm_out, state = self.lstm(lstm_in, state)

        features = lstm_out.flatten(end_dim = 1) #flattens batch and seq len dimension together
        value = self.value_head(features) #calculate the value from the lstm features

        if self.combine_discrete_moves == True:
            discrete_logits = self.discrete_action_head(features)

            if actions_taken is None:
                discrete_combo_action = self.sample_discrete(discrete_logits)
            else:
                discrete_combo_action = (actions_taken[:,3].long() * 8) + actions_taken[:,4].long()

            one_hot_discrete_action = nn.functional.one_hot(discrete_combo_action, 32)
            features_discrete = torch.cat([features, one_hot_discrete_action], axis = 1)

        else:
            focus_logits = self.focus_action_head(features)

            if actions_taken is None:
                focus_action = self.sample_discrete(focus_logits)
            else:
                focus_action = actions_taken[:,4].long()

            one_hot_focus_action = nn.functional.one_hot(focus_action, 8)
            features_discrete = torch.cat([features, one_hot_focus_action], axis = 1)

            ability_logits = self.ability_action_head(features_discrete)
            discrete_logits = [ability_logits, focus_logits]

        continuous_means = self.continuous_mean_head(features_discrete)
        continuous_stds = torch.sigmoid(self.continuous_std_head(features_discrete)) + 0.01
        #bound action 0 and 1 between -1 and 1, bound action 2 between 0 and 1
        continuous_means[:,0:2] = torch.tanh(continuous_means[:,0:2])
        continuous_means[:,2] = torch.sigmoid(continuous_means[:,2])

        if actions_taken is None:
            continuous_actions = self.sample_continuous(continuous_means, continuous_stds)

            if self.combine_discrete_moves == True:
                pass
            else:
                ability_action = self.sample_discrete(ability_logits)
                action = torch.cat([continuous_actions, ability_action.reshape(-1,1), focus_action.reshape(-1,1)], axis = 1)

            return continuous_means, discrete_logits, value, state, action

        return continuous_means, continuous_stds, discrete_logits, value, state

    def sample_discrete(self, logits):
        probs = nn.functional.log_softmax(logits, dim=1).exp()
        return torch.multinomial(probs, num_samples = 1).flatten().detach()

    def sample_continuous(self, means, stds):
        truncNorm = TruncatedNormal(self.device, means, stds, self.continuous_lower_bounds, self.continuous_upper_bounds)
        return truncNorm.rsample().detach()

    def get_action(self, obs, state):
        _, _, _, state, actions = self(obs, state)

        return actions.cpu().numpy(), state

    def get_action_info(self, continuous_means, countinuous_stds, discrete_output, actions_taken):
        truncNorm = TruncatedNormal(self.device, continuous_means, countinuous_stds, self.continuous_lower_bounds, self.continuous_upper_bounds)
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
        act = torch.Tensor(act).to(self.device)
        flattened_act = act.flatten(end_dim = 1) #flatten actions
        #useful metric that will be used several times
        fragments_per_batch = ceil(self.hyperparams["minibatch_size"] / self.hyperparams["lstm_fragment_length"])
        minibatches_before_recompute = ceil(self.hyperparams["recompute_every"] / self.hyperparams["minibatch_size"])

        continuous_means, continuous_stds, discrete_output, _, _ = self(obs, None, flattened_act)
        original_log_prob_pi, entropy = self.get_action_info(continuous_means, continuous_stds, discrete_output, flattened_act)
        original_log_prob_pi = original_log_prob_pi.detach()

        if self.dependent_movement_actions:
            original_log_prob_pi[:,:2] *= (1 - flattened_act[:,2].unsqueeze(1))

        #### DEBUGGING ####
        continuous = continuous_means.std(axis=0)
        discrete_var1 = nn.functional.log_softmax(discrete_output[0], dim=1).exp().std(axis=0).mean().item()
        discrete_var2 = nn.functional.log_softmax(discrete_output[1], dim=1).exp().std(axis=0).mean().item()

        print("Action STDs:", continuous[0].item(), continuous[1].item(), continuous[2].item(), discrete_var1, discrete_var2)

        print("Policy Entropy: ", entropy.mean(axis=0).detach().cpu().numpy().tolist())
        print("Policy Standev: ", continuous_stds.mean(axis=0).detach().cpu().numpy().tolist())
        #### DEBUGGING ####

        #break observation into fragments
        obs_fragmented = obs.reshape((obs.shape[0] * obs.shape[1]) // self.hyperparams["lstm_fragment_length"], self.hyperparams["lstm_fragment_length"], 64)

        for epoch in range(self.hyperparams['epochs_per_update']):
            shuffled = torch.randperm(obs_fragmented.shape[0])

            print("\nTraining PPO epoch", epoch)
            for minibatch_num in tqdm(range((obs_fragmented.shape[0] - 1) // fragments_per_batch)):
                #recompute advantages and hidden states periodically to keep updates accurate
                if minibatch_num % minibatches_before_recompute == 0:
                    _, _, _, frag_value, frag_state = self(obs[:, :self.hyperparams["lstm_fragment_length"], :],
                                                        None,
                                                        act[:, :self.hyperparams["lstm_fragment_length"], :].flatten(end_dim = 1))

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
                        _, _, _, frag_value, frag_state = self(obs[:, self.hyperparams["lstm_fragment_length"]*(i+1):self.hyperparams["lstm_fragment_length"]*(i+2), :],
                                                            frag_state,
                                                            act[:, self.hyperparams["lstm_fragment_length"]*(i+1):self.hyperparams["lstm_fragment_length"]*(i+2), :].flatten(end_dim = 1))

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
                shuffled_fragments = shuffled[minibatch_num*fragments_per_batch:(minibatch_num+1)*fragments_per_batch]
                fragment_indice_list = [((shuffled_fragments.unsqueeze(1) * self.hyperparams["lstm_fragment_length"]) + i) for i in range(self.hyperparams["lstm_fragment_length"])]
                shuffled_indices = torch.cat(fragment_indice_list, axis = 1).flatten()

                #get actions and normalized advantages for this minibatch
                minibatch_act = flattened_act[shuffled_indices]
                minibatch_norm_adv = torch.Tensor(norm_adv[shuffled_indices]).to(self.device)

                #initial states for this minibatch
                minibatch_state = [state[0][shuffled_fragments].to(self.device), state[1][shuffled_fragments].to(self.device)]
                continuous_means, continuous_stds, discrete_output, value, _ = self(obs_fragmented[shuffled_fragments], minibatch_state, minibatch_act)
                curr_logprob_pi, entropy = self.get_action_info(continuous_means, continuous_stds, discrete_output, minibatch_act)

                if self.dependent_movement_actions:
                    curr_logprob_pi[:,:2] *= (1 - minibatch_act[:,2].unsqueeze(1))
                    entropy[:,:2] *= (1 - minibatch_act[:,2].unsqueeze(1))

                #PPO clipped policy loss
                if self.clip_independently:
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

                entropy_loss = self.hyperparams['continuous_entropy_coeff'] * entropy[:,:3].mean() + self.hyperparams['discrete_entropy_coeff'] * entropy[:,3:].mean()

                loss = -torch.min(surrogate_loss1, surrogate_loss2).mean() - entropy_loss + self.hyperparams['value_coeff'] * value_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
