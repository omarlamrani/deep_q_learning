import keras
from keras import layers
from keras.models import Sequential


class DQN:
    """This class represents the pixel DQN"""
    def __init__(self, in_shape, actions):
        # super(DQN,self).__init__()
        self.model = Sequential()
        self.model.add(layers.Input(shape=(84,84,1, )))
        self.model.add(layers.Conv2D(32, 8, strides=4, activation="relu"))
        self.model.add(layers.Conv2D(64, 4, strides=2, activation="relu"))
        self.model.add(layers.Conv2D(64, 3, strides=1, activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dense(actions, activation="linear"))

class DQN_RAM:
    """This class represents the ram DQN"""

    def __init__(self, in_shape, actions):
        # super(DQN_RAM,self).__init__()
        self.model = Sequential()
        self.model.add(layers.Input(shape=(128,)))
        self.model.add(layers.Dense(128, activation="relu"))
        self.model.add(layers.Dense(128, activation="relu"))
        self.model.add(layers.Dense(actions, activation="linear"))