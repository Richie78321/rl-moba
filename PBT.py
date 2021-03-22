from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict
from collections.abc import Callable
from torch import nn

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

  def explore(self, exploration_methods: Dict[str, Callable[[any], any]]) -> None:
    """Explore the hyperparameter space using exploration methods that take input to the
    old hyperparameter and output the new hyperparameter.

    Args:
        exploration_methods (Dict[str, Callable[[any], any]]): The dictionary of exploration methods
          where the key is the name of the hyperparameter.
    """
    
    hyperparams_to_update = {}
    hyperparams = self.get_hyperparams()
    for hyperparam_key in exploration_methods.keys():
      if hyperparam_key not in hyperparams:
        raise ValueError("Agent does not contain the hyperparameter '{}'".format(hyperparam_key))
      
      hyperparams_to_update[hyperparam_key] = exploration_methods[hyperparam_key](hyperparams[hyperparam_key])

    self.update_hyperparams(hyperparams_to_update)