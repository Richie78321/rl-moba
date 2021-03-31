import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

with open('./hyperparam_history.json', "r") as history_file:
  hyperparam_history = json.load(history_file)

history_dicts = []
for agent_index, agent_history in enumerate(hyperparam_history):
  for iteration in agent_history.keys():
    history_dict = {}
    for hyperparam in agent_history[iteration].keys():
      history_dict[hyperparam] = agent_history[iteration][hyperparam]
    history_dict['agent'] = agent_index
    history_dict['iteration'] = iteration

    history_dicts.append(history_dict)

history_df = pd.DataFrame(history_dicts)
history_df['iteration'] = history_df['iteration'].astype(int)
history_df['agent'] = history_df['agent'].astype(int)

max_iteration = history_df['iteration'].astype(int).max()

history_df = history_df.set_index(['agent', 'iteration'])

new_index = pd.MultiIndex.from_product([range(len(hyperparam_history)), range(max_iteration + 1)], names=['agent', 'iteration'])
history_df = history_df.reindex(new_index, method='ffill')

# print(new_index)
print(history_df.sample(10))

iteration_means = history_df.groupby('iteration').mean()
print(iteration_means)

iteration_means[['discrete_entropy_coeff', 'continuous_entropy_coeff']].plot(logy=True)
plt.show()

iteration_means[['minibatch_size']].plot(logy=True)
plt.show()

iteration_means[['lstm_fragment_length']].plot(logy=True)
plt.show()
