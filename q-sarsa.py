import gym
import numpy as np

# Frozen Lake environment setup
epsilon = 0.9
env = gym.make('FrozenLake-v1')
episodes = 1000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
# We re-initialize the Q-table
qtable = np.zeros((num_states, num_actions))


# List of outcomes to plot
outcomes = []

print('Q-table before training:')
print(qtable)

# Training
for _ in range(episodes):
    state = env.reset()
    state = state[0]
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
        else:
          action = env.action_space.sample()
             
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info, _ = env.step(action)

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)



## 2nd approach

import gym
import numpy as np

# Frozen Lake environment setup
epsilon = 0.9
env = gym.make('FrozenLake-v1')
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# SARSA algorithm
for episode in range(total_episodes):
    state = env.reset()
    state = state[0]
    action = env.action_space.sample()

    for step in range(max_steps):
        # Take an action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)

        # Choose the next action using epsilon-greedy policy
        next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[next_state])

        # Update Q-value using SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action

        if done:
            break

# Print the optimal Q-values
print("Optimal Q-values:")
print(Q)
