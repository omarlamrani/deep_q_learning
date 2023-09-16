import unittest
from collections import deque

import gym
import numpy as np
from gym.wrappers import AtariPreprocessing
from tensorflow import keras
from action_obs import left_observations

import agent
import deep_Q_net
import experience_replay
import ram_explanation

env = gym.make("Breakout-ram-v4")
model = keras.models.load_model('RAM_Breakout_ep70000', compile=False)
state_size = env.observation_space.shape
action_size = env.action_space.n
net = deep_Q_net.DQN(state_size, action_size)
net.model = model
test_agent = agent.Agent(env, net, state_size, action_size)
exp = experience_replay.ReplayMemory(10)
obs_list = ram_explanation.gather_observations(env)

env_pxl = gym.make("BreakoutNoFrameskip-v4")
env_pxl = AtariPreprocessing(env_pxl, frame_skip=4, scale_obs=True)
model_pxl = keras.models.load_model('ram_model_Breakout', compile=False)
state_size_pxl = env.observation_space.shape
action_size_pxl = env.action_space.n
net_pxl = deep_Q_net.DQN(state_size, action_size)
net_pxl.model = model_pxl
test_agent_pxl = agent.Agent(env, net, state_size, action_size)


class TestAgent(unittest.TestCase):

    def test_act(self):
        test_agent.explore_rate = 0
        test_agent.state = left_observations[0]
        self.assertEqual(test_agent.act(), 3, "Should be 3")

    def test_e_decay(self):
        test_agent.explore_rate = 1
        self.assertGreaterEqual(1, test_agent.e_decay(), "Explore rate isn't lowering")


class TestDQN(unittest.TestCase):

    def test_DQN(self):
        self.assertEqual(deep_Q_net.DQN(state_size_pxl[0], action_size_pxl).model.summary(),
                         test_agent_pxl.network.model.summary())

    def test_DQN_RAM(self):
        self.assertEqual(deep_Q_net.DQN_RAM(state_size[0], action_size).model.summary(),
                         test_agent.network.model.summary())


class TestExpReplay(unittest.TestCase):
    exp.memorize(0)
    exp.memorize(1)
    exp.memorize(2)

    def test_memorize(self):
        self.assertGreaterEqual(exp.memorize(1), len(exp))

    def test_get_indices(self):
        self.assertEqual(2, len(exp.get_indices(2)))

    def test_sample(self):
        self.assertEqual(np.array([0, 2]).all(), exp.sample([0, 2]).all())


class TestRAMexp(unittest.TestCase):

    def test_gather_obs(self):
        self.assertIsInstance(deque(), type(exp.buffer))

    def test_neuron_act_values(self):
        self.assertEqual(128, len(ram_explanation.get_neuron_act_value(obs_list, net, 0)))


if __name__ == '__main__':
    unittest.main()
