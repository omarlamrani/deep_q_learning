"""This class is for testing out the trained models"""

import time
import gym
import numpy as np
import ale_py.roms as roms
import tensorflow as tf
from gym.wrappers import AtariPreprocessing
from tensorflow import keras
import argparse

gamesNames = ['Breakout', 'Pong', 'Seaquest', 'MsPacman']

# parsing arguments
parser = argparse.ArgumentParser(description="Options for testing")
parser.add_argument('-g', type=str,default='Breakout')
parser.add_argument('-p', type=str, default='pxl')
parser.add_argument('-m',type=str)
parser.add_argument('-ep',type=int,default=15)


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
# print(args.r)
GAME = args.g  # Game to learn to play in
PXL_MODE = args.p # model with RAM or pixels
MODEL = args.m # model name
EPISODES = args.ep # run for how many episodes

print(f"MODEL: {MODEL}")
print(f"GAME: {GAME}")
print(f"PXL_MODE: {PXL_MODE}")
print(f"EPISODES: {EPISODES}")

model = None
env = None

if PXL_MODE == "pxl":
    model = keras.models.load_model(MODEL, compile=False)  #pxl_model_Breakout
    env = gym.make(GAME+"NoFrameskip-v4")
    env = AtariPreprocessing(env,frame_skip=4,scale_obs=True)

else:
    model = keras.models.load_model(MODEL,compile=False) #ram_model_Breakout
    env = gym.make(GAME+'-ram-v4')

print(f"all_actions{env.observation_space}")
actions = env.action_space.n
noop_counter = 0
right_counter = 0
print(env.unwrapped.get_action_meanings())
print(actions)
print(roms.__all__)

for episode in range(0,EPISODES):
    done = False
    episode_reward = 0
    state = np.array(env.reset())

    fire_counter = 0
    while done == False:
        env.render()
        time.sleep(.02)
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        state_next, reward, done, _ = env.step(action)
        episode_reward += reward
        state = np.array(state_next)
env.close()
