"""This is our main training class
The first part represent our argument parser
The second and third part respectively represent both pixel and RAM training"""

import gym
from gym.wrappers import AtariPreprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from agent import Agent
from deep_Q_net import DQN, DQN_RAM
from experience_replay import ReplayMemory
from arg_parser import ArgParsing
import argparse
import warnings


warnings.filterwarnings("ignore")

gamesNames = ['Breakout', 'Pong', 'Seaquest', 'MsPacman']

# parsing arguments
parser = argparse.ArgumentParser(description="Options for training")
parser.add_argument('-g', type=str,default='Breakout')
parser.add_argument('-p', type=str,default='pxl')
parser.add_argument('-e', type=int, default=10000)
parser.add_argument('-r', help='Train while rendering', type=str, default="n")

args = parser.parse_args()

validGame = False

# checking whether game name is valid or not
for i in range(len(gamesNames)):
    if args.g == gamesNames[i]:
        validGame = True
if not validGame:
    print("Please relaunch the script with a valid game")
    exit(0)

# command line arguments
print(args.r)
RENDER = args.r  # Render while training
CHECKPOINT_EP = args.e  # Saving frequency
GAME = args.g  # Game to learn to play in
PXL_MODE = args.p  # Train with RAM or pixels

print(f"RENDER: {RENDER}")
print(f"CHECKPOINT_EP: {CHECKPOINT_EP}")
print(f"GAME: {GAME}")
print(f"PXL_MODE: {PXL_MODE}")

if PXL_MODE == 'pxl':

    # create Atari environment
    env = gym.make(GAME + "NoFrameskip-v4")
    # setup observation wrappers
    env = AtariPreprocessing(env, frame_skip=4, scale_obs=True)

    # all possible actions
    all_actions = env.unwrapped.get_action_meanings()
    print(all_actions)

    # get input and output sizes for network
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    print(action_size)
    print(state_size)

    # for plotting learing
    plot_rewards = []
    plot_episode = []

    # hyperparameters
    mean_episode_reward = 0
    max_exp_len = 300000
    target_update_freq = 10000
    update_freq = 4
    max_steps_per_episode = 10000
    # decaying learning rate
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=33333,
        decay_rate=0.96,
        staircase=True)

    # main network
    network = DQN(state_size, action_size)
    # target network
    target_network = DQN(state_size, action_size)
    agent = Agent(env, network, state_size, action_size)
    agent.learning_rate = lr_schedule
    epsilon = agent.explore_rate

    loss_function = tf.keras.losses.MeanSquaredError()

    mean_exp_reward = 0
    episode_count = 0
    frame_count = 0
    epochs = 0

    optimizer = tf.keras.optimizers.RMSprop()

    # experience replay buffers
    state_exp = ReplayMemory(max_len=max_exp_len)
    action_exp = ReplayMemory(max_len=max_exp_len)
    reward_exp = ReplayMemory(max_len=max_exp_len)
    next_state_exp = ReplayMemory(max_len=max_exp_len)
    done_exp = ReplayMemory(max_len=max_exp_len)
    episode_reward_exp = ReplayMemory(max_len=100)

    while True:
        try:

            # starting episode
            agent.state = env.reset()
            episode_reward = 0

            for step in range(1, max_steps_per_episode):
                if RENDER == 'y':
                    env.render()
                frame_count += 1

                # choosing best action or random
                action = agent.act()

                # acting in environment
                transition = env.step(action)

                # epsilon decay
                agent.e_decay()

                # next state after action
                next_state = transition[0]
                # adding step reward
                episode_reward += transition[1]
                # checking done for episode end
                done = transition[2]

                # memorizing transition to replay buffer
                state_exp.memorize(agent.state)
                action_exp.memorize(action)
                next_state_exp.memorize(transition[0])
                reward_exp.memorize(transition[1])
                done_exp.memorize(transition[2])

                # updating main network
                if frame_count % update_freq == 0 and len(done_exp) > agent.batch_size:
                    # indices of samples for replay buffers
                    indices = state_exp.get_indices(agent.batch_size)

                    # sample from replay buffer
                    state_sample = np.array(state_exp.sample(indices))
                    state_next_sample = np.array(next_state_exp.sample(indices))
                    rewards_sample = reward_exp.sample(indices)
                    action_sample = action_exp.sample(indices)
                    done_sample = tf.convert_to_tensor([float(i) for i in done_exp.sample(indices)])

                    # updated Q-values for the sampled future states
                    # target model avoids circling
                    with tf.device('/gpu:0'):
                        future_rewards = target_network.model.predict(state_next_sample)
                        # q value = reward + discount factor * expected future reward
                        updated_q_values = rewards_sample + agent.discount * tf.reduce_max(
                            future_rewards, axis=1
                        )
                        # # If final frame set the last value to -1
                        updated_q_values = updated_q_values * (1.0 - done_sample) - done_sample

                        # only calculate loss on the updated Q-values
                        masks = tf.one_hot(action_sample, action_size)

                    # weight adjustmente refi
                    with tf.device('/gpu:0'):
                        with tf.GradientTape() as tape:
                            # train the model on the states and updated Q-values
                            q_values = network.model(np.array(state_sample))

                            # masks to the Q-values to get the Q-value for action taken
                            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                            # loss between new Q-value and old Q-value
                            loss = loss_function(updated_q_values, q_action)
                        # backpropagation
                        grads = tape.gradient(loss, network.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, network.model.trainable_variables))

                # update the target network with new weights
                if frame_count % target_update_freq == 0:
                    epochs += 1
                    plot_episode.append(epochs)
                    plot_rewards.append(mean_exp_reward)
                    target_network.model.set_weights(network.model.get_weights())
                    print(
                        f"mean episode reward: {mean_exp_reward} at episode {episode_count}, frame count {frame_count}")
                    print(f"loss: {loss}")
                    print(f"epoch: {epochs}")
                    print(f"LR: {optimizer.learning_rate}")

                if done:
                    break

            episode_count += 1

            episode_reward_exp.memorize(episode_reward)
            mean_exp_reward = np.mean(episode_reward_exp.buffer)

            if episode_count % CHECKPOINT_EP == 0:
                network.model.save('PXL_' + GAME + '_ep' + str(episode_count))
                plt.plot(plot_episode, plot_rewards)
                plt.savefig('plot_PXL_' + GAME + '_' + str(episode_count) + '.png')

            print(f"ep: {episode_count} ; reward: {episode_reward} ; epsilon: {agent.explore_rate}")


        except KeyboardInterrupt:
            print("Ended at episode {}!".format(episode_count))
            network.model.save('TESTING' + str(episode_count))
            break

        env.close()
else:
    env = gym.make(GAME + "-ram-v4")

    all_actions = env.unwrapped.get_action_meanings()
    print(all_actions)

    state_size = env.observation_space.shape
    action_size = env.action_space.n
    print(action_size)
    print(state_size)

    plot_rewards = []
    plot_episode = []

    max_exp_len = 300000
    target_update_freq = 10000
    update_freq = 4
    max_steps_per_episode = 10000

    network = DQN_RAM(len(state_size), action_size)
    target_network = DQN_RAM(len(state_size), action_size)
    agent = Agent(env, network, state_size, action_size)

    epsilon = agent.explore_rate

    loss_function = tf.keras.losses.MeanSquaredError()

    mean_exp_reward = 0
    episode_count = 0
    frame_count = 0
    epochs = 0

    optimizer = tf.keras.optimizers.RMSprop()

    state_exp = ReplayMemory(max_len=max_exp_len)
    action_exp = ReplayMemory(max_len=max_exp_len)
    reward_exp = ReplayMemory(max_len=max_exp_len)
    next_state_exp = ReplayMemory(max_len=max_exp_len)
    done_exp = ReplayMemory(max_len=max_exp_len)
    episode_reward_exp = ReplayMemory(max_len=100)

    while True:
        try:

            agent.state = env.reset()
            episode_reward = 0

            for step in range(1, max_steps_per_episode):
                if RENDER == 'y':
                    env.render()
                frame_count += 1
                reward_N_frames = 0
                action = agent.act()

                # apply action in environment for 4 frames
                _, reward, _, _ = env.step(action)
                reward_N_frames += reward
                _, reward, _, _ = env.step(action)
                reward_N_frames += reward
                _, reward, _, _ = env.step(action)
                reward_N_frames += reward
                next_state, reward, done, _ = env.step(action)
                reward_N_frames += reward

                # sum up all 4 frames rewards
                episode_reward += reward_N_frames
                # print(agent.state.shape)
                agent.e_decay()

                # next state after action
                # checking done for episode end

                # memorizing transition to replay buffer
                state_exp.memorize(agent.state)
                action_exp.memorize(action)
                next_state_exp.memorize(next_state)
                reward_exp.memorize(reward_N_frames)
                done_exp.memorize(done)

                if frame_count % update_freq == 0 and len(done_exp) > agent.batch_size:
                    # indices of samples for replay buffers
                    indices = state_exp.get_indices(agent.batch_size)

                    # sample from replay buffer
                    state_sample = np.array(state_exp.sample(indices))
                    state_next_sample = np.array(next_state_exp.sample(indices))
                    rewards_sample = reward_exp.sample(indices)
                    action_sample = action_exp.sample(indices)
                    done_sample = tf.convert_to_tensor([float(i) for i in done_exp.sample(indices)])

                    # updated Q-values for the sampled future states
                    # target model avoids circling
                    with tf.device('/gpu:0'):
                        future_rewards = target_network.model.predict(state_next_sample)
                        # q value = reward + discount factor * expected future reward
                        updated_q_values = rewards_sample + agent.discount * tf.reduce_max(
                            future_rewards, axis=1
                        )
                        # print(done_sample)
                        # # If final frame set the last value to -1
                        updated_q_values = updated_q_values * (1.0 - done_sample) - done_sample

                        # only calculate loss on the updated Q-values
                        masks = tf.one_hot(action_sample, action_size)

                    # weight adjustmente refi
                    with tf.device('/gpu:0'):
                        with tf.GradientTape() as tape:
                            # train the model on the states and updated Q-values
                            q_values = network.model(np.array(state_sample))

                            # masks to the Q-values to get the Q-value for action taken
                            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                            # loss between new Q-value and old Q-value
                            loss = loss_function(updated_q_values, q_action)
                        # print(f"loss: {loss}")
                        # backpropagation
                        grads = tape.gradient(loss, network.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, network.model.trainable_variables))

                if frame_count % target_update_freq == 0:
                    epochs += 1
                    # update the target network with new weights
                    plot_episode.append(epochs)
                    plot_rewards.append(mean_exp_reward)
                    target_network.model.set_weights(network.model.get_weights())
                    print(
                        f"mean episode reward: {mean_exp_reward} at episode {episode_count}, frame count {frame_count}")
                    print(f"loss: {loss}")
                    print(f"epoch: {epochs}")
                    print(f"LR: {optimizer.learning_rate}")

                if done:
                    break
            episode_count += 1

            episode_reward_exp.memorize(episode_reward)
            mean_exp_reward = np.mean(episode_reward_exp.buffer)

            if episode_count % CHECKPOINT_EP == 0:
                network.model.save('RAM_' + GAME + '_ep' + str(episode_count))
                plt.plot(plot_episode, plot_rewards)
                plt.savefig('plot_RAM_' + GAME + '_' + str(episode_count) + '.png')

            print(f"ep: {episode_count} ; reward: {episode_reward} ; epsilon: {agent.explore_rate}")



        except KeyboardInterrupt:
            print("Ended at episode {}!".format(episode_count))
            network.model.save('TESTING' + str(episode_count))
            break

        env.close()
