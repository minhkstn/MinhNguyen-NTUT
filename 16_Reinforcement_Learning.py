import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math

from scipy.ndimage import interpolation

import reinforcement_learning as rl

print("OpenAI Gym Version:", gym.__version__)

###############################################################################
# Game environment
env_name = 'Breakout-v0' # the name of the game-environment
rl.checkpoint_base_dir = 'checkpoints_tutorial16/'
rl.update_paths(env_name=env_name) # to set all the paths that will be used

###############################################################################
# Create agent
"""
create an object-instance
training=True: want to use the replay-mem to record states and Q-values for plotting further below
use_logging=False: don't corrupt the logs from the actual training that was done previously.
render=True: no effect
"""
agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=True)

model = agent.model
# The replay-mem require > 3GB of RAM
replay_memory = agent.replay_memory
###############################################################################
# Training
agent.run(num_episodes=1000)   # to play the game
"""
return: 1:357	 Epsilon: 1.00	 Reward: 3.0	 Episode Mean: 3.0
1: the number of episodes that have been processed
357: the number of states that have been processed
Both are stored in the TensorFlow checkpoint
"""

# Training progress
# write and read logs
log_q_values = rl.LogQValues()
log_reward = rl.LogReward()

log_q_values.read()
log_reward.read()

# Reward
# Show the reward for each episode during training + the running mean of the last 30 episodes
plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-count for Game Environment')
plt.legend()
plt.show()

# Q-values
"""
The fast improvement in the beginning is probably due to
    (1) the use of a smaller replay-memory early in training so the Neural Network
     is optimized more often and the new information is used faster
    (2) the backwards-sweeping of the replay-memory so the rewards are used to
     update the Q-values for many of the states, instead of just updating the
     Q-values for a single state
    (3) the replay-memory is balanced so at least half of each mini-batch contains
     states whose Q-values have high estimation-errors for the Neural Network
"""
plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()
##################################################################################
# Testing
print("Epsilon: ", agent.epsilon_greedy.epsilon_testing)
no longer perform training
agent.training = False

# reset the previous episode rewards
agent.reset_episode_rewards()

# render the game-environment to screen ==> see the agent playing the game
agent.render = True

agent.run(num_episodes=1)

# Mean Reward
# reset the previous episode rewards
agent.reset_episode_rewards()
agent.render = False
agent.run(num_episodes=30)

#################################################################################
# print some statistics for the episode rewards
rewards = agent.episode_rewards
print("Rewards for {0} episodes: ".format(len(rewards)))
print("- Min: ", np.min(rewards))
print("- Mean: ", np.mean(rewards))
print("- Max: ", np.max(rewards))
print("- Stdev: ", np.std(rewards))

_ = plt.hist(rewards, bins=30)
#################################################################################
# Example states
# print the Q-values for a given index in the replay-memory
def print_q_values(idx):
    q_values = replay_memory.q_values[idx]
    action = replay_memory.actions[idx]

    print("Action:      Q-values:")
    print("======================")

    # print all the actions and their Q-values
    for i, q_value in enumerate(q_values):
        if i == action:
            action_taken = "(Action Taken)"
        else:
            action_taken = ""

        action_name = agent.get_action_name(i)
        print("{0:12}{1:.3f} {2}".format(action_name, q_value, action_taken))
    print()

# plot a state from the replay-memory and optionally prints the Q-values
def plot_state(idx, print_q=True):
    state = replay_memory.states[idx]

    fig, axes = plt.subplots(1, 2)

    ax = axes.flat[0]
    ax.imshow(state[:,:,0], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')

    # plot the motion-trace
    ax = axes.flat[1]
    ax.imshow(state[:,:,0], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')

    plt.show()

    if print_q:
        print_q_values(idx=idx)

num_used = replay_memory.num_used
print("Num used: ", num_used)

# get the Q-values from the replay-mem that are actually used
q_values = replay_memory.q_values[0:num_used, :]
q_values_min = q_values.min(axis=1)
q_values_max = q_values.max(axis=1)
q_values_diff = q_values_max - q_values_min

# Example states: Highest Reward
idx = np.argmax(replay_memory.rewards)
print("The index of the highest reward: ", idx)

for i in range(-5, 3):
    plot_state(idx=idx+i)

# Example: Highest Q-value
idx = np.argmax(q_values_max)
print("The index of the highest Q-value: ", idx)

for i in range(0, 5):
    plot_state(idx=idx+i)

# Example: Loss of life
idx = np.argmax(replay_memory.end_life)
print("The state leading up to a loss of life: ", idx)
for i in range(-10, 0):
    plot_state(idx=idx+i)

# Example: Greatest Difference in Q-Values
idx = np.argmax(q_values_diff)
print("The state having the greatese difference in Q-values: ", idx)

for i in range (0, 5):
    plot_state(idx=idx+i)

############################################################################
# Output of Conv Layers
def plot_layer_output(model, layer_name, state_index, inverse_cmap=False):
    """

    :param model: an instance of the Neural Network class
    :param layer_name: name of the conv layeer
    :param state_index: index into the replay-mem for a state that will be
                        input to the NN
    :param inverse_cmap: boolean whether to inverse the color map
    """
    # get the given state-array from the replay-memory
    state = replay_memory.states[state_index]
    # get the output tensor
    layer_tensor = model.get_layer_tensor(layer_name=layer_name)
    # get the actual values of the tensor
    values = model.get_tensor_value(tensor=layer_tensor, state=state)
    # number of image channels
    num_images = values.shape[3]

    num_grids = math.ceil(math.sqrt(num_images))

    fig, axes = plt.subplots(num_grids, num_grids, figsize=(10,10))
    print("Dim, of each image: ", values.shape)

    if inverse_cmap:
        cmap = 'gray_r'
    else:
        cmap = 'gray'

    # plot the outputs of all the channels in the conv-layer
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # get the image for the i'th output channel
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap=cmap)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

################################################################################
# game state
idx = np.argmax(q_values_max)
plot_state(idx=idx, print_q=False)

# output of Conv layer 1
plot_layer_output(model=model, layer_name='layer_conv1', state_index=idx, inverse_cmap=False)
# output of Conv layer 2
plot_layer_output(model=model, layer_name='layer_conv2', state_index=idx, inverse_cmap=False)
# output of Conv layer 3
plot_layer_output(model=model, layer_name='layer_conv3', state_index=idx, inverse_cmap=False)

# Weights for Conv Layers
def plot_conv_weights(model, layer_name, input_channel=0):
    weights_variable = model.get_weights_variable(layer_name=layer_name)

    # w: 4-dim tensor
    w = model.get_variable_value(variable=weights_variable)
    # get the weights for the given channel
    w_channel = w[:, :, input_channel, :]

    num_output_channels = w_channel.shape[2]

    w_min = np.min(w_channel)
    w_max = np.max(w_channel)

    abs_max = max(abs(w_max), abs(w_min))

    print("Min:     {0:.5f}, Max:       {1:.5f}".format(w_min, w_max))
    print("Mean:    {0:.5f}, Stdev:     {1:.5f}".format(w_channel.mean(), w_channel.std()))

    num_grids = math.ceil(math.sqrt(num_output_channels))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_output_channels:
            # get the weights of the i'th filter
            img = w_channel[:, :, i]

            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=0)
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=1)

plot_conv_weights(model=model, layer_name='layer_conv2', input_channel=0)

plot_conv_weights(model=model, layer_name='layer_conv3', input_channel=0)