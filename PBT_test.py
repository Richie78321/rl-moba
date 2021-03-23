from PBT import *
from torch import nn
import torch
import random

class TestPBTAgent(PBTAgent):
  def __init__(self, hyperparams):
    super().__init__()

    self.hyperparams = hyperparams
    self.layer = nn.Linear(10, 1)

  def get_hyperparams(self):
    return self.hyperparams

  def update_hyperparams(self, hyperparams_changed):
    for hyperparam_key in hyperparams_changed.keys():
      self.hyperparams[hyperparam_key] = hyperparams_changed[hyperparam_key]

agents = [
  TestPBTAgent({
    'param1': 0,
    'param2': 1, 
  }),
  TestPBTAgent({
    'param1': 2,
    'param2': 3, 
  }),
  TestPBTAgent({
    'param1': 4,
    'param2': 5, 
  }),
  TestPBTAgent({
    'param1': 6,
    'param2': 7, 
  }),
]

rewards = [
  1,
  2,
  3,
  4
]

random.seed(78321)
perturb_explore = get_perturb_explore()

pbt_update(list(zip(agents, rewards)), 
  exploit_methods={
    'param1': lambda x: x,
    'param2': lambda x: x,
  }, 
  explore_methods={
    'param1': perturb_explore,
    'param2': perturb_explore,
  },
  exploit_portion=0.5)