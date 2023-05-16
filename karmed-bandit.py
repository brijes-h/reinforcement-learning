import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(epsilon, q_values):
    if np.random.random() < epsilon:
        # Exploration: Choose a random action
        action = np.random.randint(len(q_values))
        print('action in epsilon_greedy (exploration): ',action)
    else:
        # Exploitation: Choose the action with the highest Q-value
        action = np.argmax(q_values)
        print('action in epsilon_greedy (exploitation): ',action)
    return action

def run_bandit(epsilon):
    num_arms = 10
    num_episodes = 1000
    rewards = np.zeros(num_episodes)
    q_values = np.zeros(num_arms)
    action_counts = np.zeros(num_arms)

    for episode in range(num_episodes):
        action = epsilon_greedy(epsilon, q_values)
        reward = np.random.normal(0, 1)  # Reward from a normal distribution with mean 0
        rewards[episode] = reward

        # Update action value estimates
        action_counts[action] += 1
        q_values[action] += (reward - q_values[action]) / action_counts[action]
        '''print("episode = ",episode)
        print('action: ',action)
        print('reward: ',reward)
        print('rewards: ',rewards)
        print('action_counts: ',action_counts)
        print('q_values: ',q_values)
        print('----------------------------------------------------------------------------------------------')'''

    return rewards

# Run bandit problem for different epsilon values
epsilon_values = [0, 0.1, 0.01]
results = []

for epsilon in epsilon_values:
    rewards = run_bandit(epsilon)
    results.append(rewards)

# Plotting
plt.figure(figsize=(10, 6))
for i, epsilon in enumerate(epsilon_values):
    plt.plot(range(len(results[i])), results[i], label=f'epsilon={epsilon}')

plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward Comparison')
plt.legend()
plt.show()




# 2nd and better approacch
import numpy as np
import matplotlib.pyplot as plt

# Define the number of arms and the number of episodes
num_arms = 10
num_episodes = 1000

# Define the epsilon values to test
epsilons = [0, 0.1, 0.01]

# Define the true reward distribution for each arm
reward_means = np.random.normal(loc=0, scale=1, size=num_arms)

# Initialize the estimated reward distribution for each arm
estimated_means = np.zeros(num_arms)

# Initialize the number of times each arm has been pulled
num_pulls = np.zeros(num_arms)

# Define the epsilon-greedy action selection function
def epsilon_greedy(epsilon):
    if np.random.uniform() < epsilon:
        # Choose a random arm
        action = np.random.choice(num_arms)
    else:
        # Choose the arm with the highest estimated mean reward
        action = np.argmax(estimated_means)
    return action

# Initialize arrays to store the rewards and average rewards for each episode
rewards = np.zeros((len(epsilons), num_episodes))
avg_rewards = np.zeros((len(epsilons), num_episodes))

# Loop over the episodes
for i in range(num_episodes):
    # Loop over the epsilon values
    for j, epsilon in enumerate(epsilons):
        # Choose an action using the epsilon-greedy method
        action = epsilon_greedy(epsilon)

        # Pull the arm and observe the rewar
        reward = np.random.normal(loc=reward_means[action], scale=1)

        # Update the estimated mean reward for the chosen arm
        num_pulls[action] += 1
        estimated_means[action] += (reward - estimated_means[action]) / num_pulls[action]

        # Store the reward and average reward
        rewards[j, i] = reward
        avg_rewards[j, i] = np.mean(rewards[j, :i+1])

# Plot the average rewards for each epsilon value
for j, epsilon in enumerate(epsilons):
    plt.plot(avg_rewards[j, :], label='epsilon = ' + str(epsilon))
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.show()

     

