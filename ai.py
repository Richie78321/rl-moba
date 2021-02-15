from gym_derk.envs import DerkEnv
import numpy as np

env = DerkEnv(mode="connected", home_team=[ { 'primaryColor': '#ff00ff', 'slots': ['BloodClaws', 'IronBubblegum', 'HealingGland'] }, { 'primaryColor': '#00ff00', 'slots': ['Blaster', 'HeliumBubblegum', 'ParalyzingDart'] }, { 'primaryColor': '#ff0000', 'slots': ['Cripplers', 'Shell', 'VampireGland']}])
for t in range(2):
  observation_n = env.reset()


  # Initialize our values
  healerActions=env.action_space.sample()
  rangedActions=env.action_space.sample()
  tankActions=env.action_space.sample()
  
  while True:

    action_n = [healerActions, rangedActions, tankActions]
    observation_n, reward_n, done_n, info = env.step(action_n)
    
    healerObs=observation_n[0, :]
    rangedObs=observation_n[1, :]
    tankObs=observation_n[2, :]

    
    # Healer Actions
    # If nobody is badly injured, attack
    



    # Ranged Actions

    # Tank Actions
    
    if all(done_n):
      print("Episode finished")
      break
env.close()