import numpy as np
import gym

env = gym.make("MountainCar-v0")

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

# Hyperparameters
LEARNING_RATE = 0.8
DISCOUNT = 0.95
EPISODES = 25000 # Total Episodes to run
SHOW_EVERY = 1000 # Render every # of trys
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
epsilon = 0.8
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_devay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#print(discrete_os_win_size)

# Q-Table initialization
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Continious to Discrete transformation of position and velocity values
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

# Main Algorithm
for episode in range(EPISODES):
    # Render every few runs
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    
    discrete_state = get_discrete_state(env.reset()) # Getting initial state as discrete value

    done = False
    while not done:
        # Allowing agent to try new ways apart from determined approaches to explore new ways
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action) # Action is taken
        new_discrete_state = get_discrete_state(new_state) # New state is obtained as discrete

        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) # Max expected value in next step
            current_q = q_table[discrete_state + (action, )] # Selection of current Q value

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) # Learning alghorithm
            q_table[discrete_state+(action, )] = new_q # Updating Q-table
        elif new_state[0] > env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
    
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_devay_value
env.close()
