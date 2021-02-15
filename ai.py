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
    # If nobody is badly injured or health is too low to heal, attack
    # If somebody is injured, run to safety and heal them



    # Ranged Actions
    # If an enemy is too close, back up
    # Target enemies with healing first, then big damage items, then the closest enemy

    # Tank Actions
    # Engage the ranged units while staying close to the healer
    
    if all(done_n):
      print("Episode finished")
      break
env.close()