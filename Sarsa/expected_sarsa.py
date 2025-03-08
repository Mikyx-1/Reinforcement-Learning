### Waiting for implementation
import gym
import numpy as np

# Create the Taxi environment
env = gym.make("Taxi-v3")

# SARSA hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_epsidoes = 100000

# Initialise Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# SARSA algorithm
for episode in range(num_epsidoes):
    state = env.reset()[0]
    done = False
    
    # Choose action using epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
        
    
    total_rewards = 0
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Choose next action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[state])

        expected_value = (1 - epsilon) * np.max(Q[next_state]) + epsilon * np.mean(Q[next_state]) # Expected value in both exploitation and exploration mode
        Q[state, action] += alpha * (reward + gamma * expected_value - Q[state, action])
        
        state = next_state
        action = next_action
        total_rewards += reward

    if (episode+1)%100 == 0:
        print(f"Episode: {episode}, Total rewards: {total_rewards}")
        

        


