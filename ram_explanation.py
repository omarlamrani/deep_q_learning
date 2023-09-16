"""This class isn't very functional it can be run to obtain the same graphs as in th report"""

from action_obs import right_observations, left_observations
from collections import deque
import gym
import tensorflow as tf
from tensorflow import keras

# loading env + model
env = gym.make('Breakout-ram-v4')
env.reset()
model = keras.models.load_model('RAM_Breakout_ep70000', compile=False)


def gather_observations(environ):
    """This method gather observations from a single episodes, to be exploited
    along with the RAM"""
    obs_list = deque()
    for episode in range(1):
        right_counter = 1
        left_counter = 1
        fire_counter = 1
        observation = environ.reset()
        done = False
        episode_reward = 0
        while done == False:
            state_tensor = tf.convert_to_tensor(observation)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
            # if action == 1:
            #     environ.env.ale.saveScreenPNG('fire2_' + str(fire_counter) + '.png')
            #     print("fire" + str(fire_counter))
            #     print(['fire', observation])
            #     fire_counter += 1
            # if action == 3:
            #     environ.env.ale.saveScreenPNG('left2_' + str(left_counter) + '.png')
            #     print("left" + str(left_counter))
            #     print(['left', observation])
            #     left_counter += 1
            # if action == 2:
            #     environ.env.ale.saveScreenPNG('right2_' + str(right_counter) + '.png')
            #     print("right" + str(right_counter))
            #     print(['right', observation])
            #     right_counter += 1
            observation, reward, done, info = environ.step(action)
            episode_reward += reward
            obs_list.append(observation)
        return obs_list

def get_neuron_act_value(obs, net, neuron):
    """This method calculate all neuron-cell combinations for one neuron"""
    pixel1 = []
    for i in range(128):
        result = obs[i] \
                 * model.layers[1].get_weights()[0][neuron][i]
        pixel1.append(result)

    return pixel1

def plot_act_values_RAM():
    """This method plot the gathered activation values"""
    for state_RAM in left_observations:
        all_neurons = []

        for x in range(128):
            # print(f"neuron: {x}")
            all_neurons.append(get_neuron_act_value(state_RAM, model, x))

        import numpy as np
        import seaborn as sns
        import matplotlib.pylab as plt

        new_vals = []
        for neurons in all_neurons:
            new_row = []
            # scaling down the values
            for x in neurons:
                if x >= 0:
                    new_row.append(255 * x / 1236.6093)
                else:
                    new_row.append(255 * x / 1574.3721)
            new_vals.append(new_row)

        uniform_data = np.clip(np.array(new_vals), -14, 14)
        ax = sns.heatmap(uniform_data, linewidth=0.5, cmap=sns.diverging_palette(240, 10, n=15), cbar=False)
        print("saved")
        plt.savefig('left_' + str(left_observations.index(state_RAM)) + '_plot.png')

plot_act_values_RAM()