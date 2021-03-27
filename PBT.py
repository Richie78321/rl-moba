from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from collections.abc import Callable
from torch import nn
import numpy as np
from numpy import random
from math import floor, ceil
import random as pyrandom

class PBTAgent(ABC, nn.Module):
  @abstractmethod
  def get_hyperparams(self) -> Dict[str, any]:
    pass

  @abstractmethod
  def update_hyperparams(self, hyperparams_changed: Dict[str, any], evoke_update_event: bool = True) -> None:
    pass

  def exploit(self, other_agent: PBTAgent, exploit_methods: Dict[str, Callable[[any], any]], evoke_update_event: bool = True) -> None:
    """Exploit another agent by copying its hyperparameters and current model parameters.

    Args:
        other_agent (PBTAgent): The agent to exploit.
        exploit_methods (Dict[str, Callable[[any], any]])): The dictionary of exploitation methods where the key is
          the name of the hyperparameter.
        evoke_update_event (bool) Whether or not to evoke the hyperparameter update event. When in doubt, keep it true.
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

    self.update_hyperparams(hyperparams_to_update, evoke_update_event)

  def explore(self, explore_methods: Dict[str, Callable[[any], any]], evoke_update_event: bool = True) -> None:
    """Explore the hyperparameter space using exploration methods that take input to the
    old hyperparameter and output the new hyperparameter.

    Args:
        explore_methods (Dict[str, Callable[[any], any]]): The dictionary of exploration methods
          where the key is the name of the hyperparameter.
        evoke_update_event (bool) Whether or not to evoke the hyperparameter update event. When in doubt, keep it true.
    """
    
    hyperparams_to_update = {}
    hyperparams = self.get_hyperparams()
    for hyperparam_key in explore_methods.keys():
      if hyperparam_key not in hyperparams:
        raise ValueError("Agent does not contain the hyperparameter '{}'".format(hyperparam_key))
      
      hyperparams_to_update[hyperparam_key] = explore_methods[hyperparam_key](hyperparams[hyperparam_key])

    self.update_hyperparams(hyperparams_to_update, evoke_update_event)

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

def pbt_update_all(agents_and_rewards: List[Tuple[PBTAgent, float, bool]], exploit_methods: Dict[str, Callable[[any], any]], explore_methods: Dict[str, Callable[[any], any]], exploit_portion: float = 0.4) -> List[int]:
  """Update the PBT agents using the accumulated rewards as a judgement of fitness. The top agents
  based on cumulative reward that are in the exploit portion are exploited by the agents that are not
  in the exploit portion.

  Args:
      agents_and_rewards (List[Tuple[PBTAgent, float, bool]]): A list of tuples containing the PBT agent to update,
        the accumulated reward for that agent since the last update, and whether or not that agent is
        ready to update.
      exploit_methods (Dict[str, Callable[[any], any]]): A dictionary of exploitation methods where
        the key is the name of the hyperparameter.
      explore_methods (Dict[str, Callable[[any], any]]): The dictionary of exploration methods
          where the key is the name of the hyperparameter.
      exploit_portion (float, optional): A value from 0.0 to 1.0 that determines what portion of
        fittest agents should be chosen from for exploitation. Defaults to 0.4.

    Returns:
        List[int]: Returns the list of indices of agents updated.
  """
  if exploit_portion < 0 or exploit_portion > 1:
    raise ValueError("Invalid exploit portion. Must be between 0 and 1.") 
  
  agents_and_rewards_sorted = np.flip(np.argsort([x[1] for x in agents_and_rewards]))
  num_to_exploit = int(ceil(len(agents_and_rewards) * exploit_portion))
  
  exploited_agents = [agents_and_rewards[x][0] for x in agents_and_rewards_sorted[:num_to_exploit]]
  exploiter_agents_indices = [x for x in filter(lambda x: agents_and_rewards[x][2], agents_and_rewards_sorted[num_to_exploit:])]

  for agent_index in exploiter_agents_indices:
    # Don't evoke update event because running explore directly after exploit.
    agents_and_rewards[agent_index][0].exploit(random.choice(exploited_agents), exploit_methods, evoke_update_event=False)
    agents_and_rewards[agent_index][0].explore(explore_methods)

  return exploiter_agents_indices

def pbt_update_bottom(agents_and_rewards: List[Tuple[PBTAgent, float, bool]], exploit_methods: Dict[str, Callable[[any], any]], explore_methods: Dict[str, Callable[[any], any]], exploit_portion: float = 0.2, exploiter_portion: float = 0.2) -> None:
  """Update the PBT agents using the accumulated rewards as a judgement of fitness. The top agents
  based on cumulative reward that are in the exploit portion are exploited 

  Args:
      agents_and_rewards (List[Tuple[PBTAgent, float, bool]]): A list of tuples containing the PBT agent to update,
        the accumulated reward for that agent since the last update, and whether or not that agent is
        ready to update.
      exploit_methods (Dict[str, Callable[[any], any]]): A dictionary of exploitation methods where
        the key is the name of the hyperparameter.
      explore_methods (Dict[str, Callable[[any], any]]): The dictionary of exploration methods
          where the key is the name of the hyperparameter.
      exploit_portion (float, optional): Determines what portion of
        fittest agents should be chosen from for exploitation. Defaults to 0.2.
      exploiter_portion (float, optional): Determines what portion of least fit agents should
        be chosen to exploit. Defaults to 0.2.
  """
  if exploit_portion < 0 or exploiter_portion < 0 or exploit_portion + exploiter_portion > 1:
    raise ValueError("Invalid exploit and exploiter portions. Each must be between 0 and 1, and their sum must not be greater than 1.")

  agents_and_rewards_sorted = np.flip(np.argsort([x[1] for x in agents_and_rewards]))
  num_to_exploit = int(ceil(len(agents_and_rewards) * exploit_portion))
  num_exploiters = int(floor(len(agents_and_rewards) * exploiter_portion))
  
  exploited_agents = [agents_and_rewards[x][0] for x in agents_and_rewards_sorted[:num_to_exploit]]
  exploiter_agents_indices = [x for x in filter(lambda x: agents_and_rewards[x][2], agents_and_rewards_sorted[-num_exploiters:])]

  for agent_index in exploiter_agents_indices:
    # Don't evoke update event because running explore directly after exploit.
    agents_and_rewards[agent_index][0].exploit(random.choice(exploited_agents), exploit_methods, evoke_update_event=False)
    agents_and_rewards[agent_index][0].explore(explore_methods)

  return exploiter_agents_indices