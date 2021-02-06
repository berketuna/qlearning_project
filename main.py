import numpy as np
import gym

env = gym.make("MountainCar-v0")
env.reset()

done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()
env.close()

'''
env = gym.make(env.observation_space.high)
env = gym.make(env.observation_space.low)
env = gym.make(env.action_space.n)
'''
