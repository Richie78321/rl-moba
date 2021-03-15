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


class lstm_agent(nn.Module):
    def __init__(self, lstm_size, device, activation = nn.Tanh()):
        super().__init__()
        self.device = device
        self.ELO=1000

        #### HYPERPARAMETERS ####

        self.learning_rate = 2e-3
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
        self.lstm_fragment_length = 15
        #batch size in fragments
        self.fragments_per_batch = 900 // self.lstm_fragment_length
        #how many times to loop over the entire batch of experience
        self.epochs_per_update = 1
        # how often to recompute hidden states and advantages. Expensive but allows more accurate training
        self.recompute_every = 20
        # defines size of trust region, smaller generally means more stable but slower learning
        self.eps_clip = 0.2
        # how much to optimize for entropy, prevents policy collapse by keeping some randomness for exploration
        self.entropy_coeff = 0.004
        # whether to treat each component of an action as independent or not.
        # by default this should be set to False
        self.independent_action_components = False

        #### ARCHITECTURE ####

        self.lstm = nn.LSTM(64, lstm_size, batch_first = True)

        self.continuous_size = 3
        self.logstd = nn.Parameter(torch.Tensor([0.693147, 0.693147, 0]))
        self.continuous_lower_bounds = torch.Tensor([-1, -1, 0]).to(self.device)
        self.continuous_upper_bounds = torch.Tensor([1, 1, 1]).to(self.device)
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



"""
Plan

Assign games based on policy vs policy instead of team vs team

Run multiple iterations of each game and pick the winner based on win %

Adjust ELO for each team accordingly

Intelligently assign the next matchup

"""


device = "cuda:0"
ITERATIONS = 1000000
root_dir="checkpoints/TEST_LEAGUE_AGENTS"
league=[]
for root, dirs, files in os.walk(root_dir):
    for name in files: # Load in the agents stored at root_dir
        temp=lstm_agent(512, device)
        temp.load_state_dict(torch.load(os.path.join(root, name)))
        league.append(temp)

league_size = len(league) # Number of policies.  Must be even because we don't want byes or anything like that.
teams_per_member=5 # Number of teams per policy.  Must be odd so there are no ties.  Best of x.
assert league_size%2 == 0, "Number of policies in the TEST_LEAGUE_AGENTS folder must be even"
assert teams%2 == 1, "Number of teams per policy must be odd"

env = DerkEnv(n_arenas = (league_size*teams_per_member)/2, turbo_mode = True, reward_function = win_loss_reward_function, home_team = classes_team_config, away_team = classes_team_config)

for iteration in range(ITERATIONS):
    print("\n-----------------------------ITERATION " + str(iteration) + "-----------------------------")

    #randomize matchings between policies to start (This will eventually be more intelligent) - Can just sort by ELO and then assign that way
    policy_matchup_IDS=np.random.permutation(league_size)

    observation = [[] for i in range(league_size)]
    action = [[] for i in range(league_size)]
    reward = [[] for i in range(league_size)]
    states = [None for i in range(league_size)]

    observation_n = env.reset()
    with torch.no_grad():
        while True:
            action_n = np.zeros((env.n_agents, 5)) # Array of 0's that is the right size
            for i in range(league_size):
                action_n[list(range(teams_per_member*i*3, teams_per_member*(i+1)*3))], states[i] = league[policy_matchup_IDS[i]].get_action(observation_n[list(range(teams_per_member*i*3, teams_per_member*(i+1)*3))], states[i]) # league is where the policies are stored

            #act in environment and observe the new obervation and reward (done tells you if episode is over)
            observation_n, reward_n, done_n, _ = env.step(action_n)

            if all(done_n): # We have to update ELO here
              break

env.close()

