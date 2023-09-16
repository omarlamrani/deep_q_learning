import random
from collections import deque, namedtuple
import numpy as np


# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done','info'))

class ReplayMemory:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def memorize(self, element):
        self.buffer.append(element)
        return len(self)

    def get_indices(self, batch_size):
        indices = np.random.choice(range(self.__len__()), size=batch_size)
        return indices

    def sample(self, indices):
        batch = np.array([self.buffer[i] for i in indices])
        return batch

    def __len__(self):
        return len(self.buffer)


