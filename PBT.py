from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from collections.abc import Callable
from torch import nn
from numpy import random
from math import ceil
import random as pyrandom

class PBTAgent(ABC, nn.Module):
  @abstractmethod
  def get_hyperparams(self) -> Dict[str, any]:
    pass

  @abstractmethod
  def update_hyperparams(self, hyperparams_changed: Dict[str, any]) -> None:
    pass

  def exploit(self, other_agent: PBTAgent, exploit_methods: Dict[str, Callable[[any], any]]) -> None:
    """Exploit another agent by copying its hyperparameters and current model parameters.

    Args:
        other_agent (PBTAgent): The agent to exploit.
        exploit_methods (Dict[str, Callable[[any], any]])): The dictionary of exploitation methods where the key is
          the name of the hyperparameter.
    """
    # Copy module parameters
    self.load_state_dict(other_agent.state_dict())
    
    # Copy hyperparameters
    hyperparams_to_update = {}
    other_agent_hyperparams = other_agent.get_hyperparams()
    for hyperparam_key in exploit_methods.keys():
      if hyperparam_key not in other_agent_hyperparams:
        raise ValueError("Other agent does contain the hyperparameter '{}'".format(hyperparam_key))
      
      hyperparams_to_update[hyperparam_key] = exploit_methods[hyperparam_key](other_agent_hyperparams[hyperparam_key])

    self.update_hyperparams(hyperparams_to_update)

  def explore(self, explore_methods: Dict[str, Callable[[any], any]]) -> None:
    """Explore the hyperparameter space using exploration methods that take input to the
    old hyperparameter and output the new hyperparameter.

    Args:
        explore_methods (Dict[str, Callable[[any], any]]): The dictionary of exploration methods
          where the key is the name of the hyperparameter.
    """
    
    hyperparams_to_update = {}
    hyperparams = self.get_hyperparams()
    for hyperparam_key in explore_methods.keys():
      if hyperparam_key not in hyperparams:
        raise ValueError("Agent does not contain the hyperparameter '{}'".format(hyperparam_key))
      
      hyperparams_to_update[hyperparam_key] = explore_methods[hyperparam_key](hyperparams[hyperparam_key])

    self.update_hyperparams(hyperparams_to_update)

def get_perturb_explore(perturbation_factor: float = 0.2) -> Callable[[float], float]:
  """Get a perturb exploration method using the given perturbation factor.

  Args:
      perturbation_factor (float, optional): 1 plus or minus this factor is multiplied onto 
        the value. Defaults to 0.2.

  Returns:
      Callable[[float], float]: Returns the perturbation exploration function for this perturbation
        factor.
  """
  
  def perturb_explore(value: float) -> float:
    if bool(pyrandom.getrandbits(1)):
      value *= 1 - perturbation_factor
    else:
      value *= 1 + perturbation_factor

    return value

  return perturb_explore

def pbt_update(agents_and_rewards: List[Tuple[PBTAgent, float]], exploit_methods: Dict[str, Callable[[any], any]], explore_methods: Dict[str, Callable[[any], any]], exploit_portion: float = 0.4) -> None:
  """Update the PBT agents using the accumulated rewards as a judgement of fitness.

  Args:
      agents_and_rewards (List[Tuple[PBTAgent, float]]): A list of tuples containing the PBT agent to update 
        and the accumulated reward for that agent since the last update.
      exploit_methods (Dict[str, Callable[[any], any]]): A dictionary of exploitation methods where
        the key is the name of the hyperparameter.
      explore_methods (Dict[str, Callable[[any], any]]): The dictionary of exploration methods
          where the key is the name of the hyperparameter.
      exploit_portion (float, optional): A value from 0.0 to 1.0 that determines what portion of
        fittest agents should be chosen from for exploitation. Defaults to 0.4.
  """
  
  agents_and_rewards.sort(reverse=True, key=lambda x: x[1])
  num_to_exploit = int(ceil(len(agents_and_rewards) * exploit_portion))
  
  exploited_agents = [x[0] for x in agents_and_rewards[:num_to_exploit]]
  exploiter_agents = [x[0] for x in agents_and_rewards[num_to_exploit:]]

  for agent in exploiter_agents:
    agent.exploit(random.choice(exploited_agents), exploit_methods)
    agent.explore(explore_methods)