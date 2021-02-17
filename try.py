from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import gym
from gym.envs.toy_text.frozen_lake import LEFT, DOWN, RIGHT, UP

# ********************
# TODO Create environment
env = gym.make("FrozenLake-v0")


# TODO Render initial state
env.render()

# ********************

Q = np.random.uniform(low=-2, high=-1, size = (16,4))

# ********************

# Hyperparameters
lr = 0.9
gamma = 0.95
num_episodes = 1000
max_steps_per_episode = 100
#possible_actions = [LEFT, DOWN, RIGHT, UP]

# Create lists to contain total rewards and steps per episode
rewards = np.zeros(num_episodes)

for i in tqdm(range(num_episodes), desc="Training"):
    # Reset environment and observe initial state
    s = env.reset()

    # The Q-Table learning algorithm
    for _ in range(max_steps_per_episode):
        # ********************
        # TODO Choose action
        a = np.argmax(Q[s]) # LEFT,DOWN,RIGHT,UP
                            # 0   ,1   ,2    ,3

        # ********************

        # Get new state and reward from environment
        s_new, r, done, _ = env.step(a)

        # ********************
        # TODO Update Q-table
        Q[s, a] = (1-lr) * Q[s, a] + lr*(r + gamma*np.max(Q[s_new]))

        # ********************

        # Bookkeeping
        rewards[i] += r
        s = s_new

        if done:  # Check if episode terminated
            break    

# Plot rewards
sns.lineplot(data=pd.DataFrame(rewards, columns=["training"]).rolling(50).mean())
plt.xlabel("epoch")
plt.ylabel("reward")
plt.tight_layout()
plt.savefig("solution/a1c-train.png")