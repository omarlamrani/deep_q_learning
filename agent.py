import numpy as np
import tensorflow as tf

class Agent:

    def __init__(self, env, net, in_shape, out_shape):
        """This the constructor class for the agent, contains all necessary hyperparameters"""
        self.env = env
        # self.exp = exp
        self.state = env.reset()
        self.total_reward = 0.0
        self.state_size = in_shape
        self.action_size = out_shape
        self.discount = 0.99 # discount rate
        self.explore_rate = 1.0  # exploration rate
        self.explore_max = 1
        self.explore_min = 0.05
        self.explore_decay_frames = 1000000
        self.learning_rate = 0.00015
        self.batch_size = 32
        self.network = net
        self.start_count = 0

    def act(self):
        """This class returns either a random action or the max value
        predicted by the network"""
        if self.explore_rate > np.random.rand(1)[0]:
            action = np.random.choice(self.action_size)
            return action
        else:
            state_tensor = tf.convert_to_tensor(self.state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.network.model(state_tensor, training=False)
            return np.argmax(action_probs[0])

    def e_decay(self):
        """"This class decays the epsilon/explore rate every timestep
        until it reaches its minimum value"""
        self.explore_rate -= (self.explore_max - self.explore_min) / self.explore_decay_frames
        self.explore_rate = max(self.explore_rate, self.explore_min)
        return self.explore_rate
