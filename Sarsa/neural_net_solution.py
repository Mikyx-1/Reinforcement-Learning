import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Create the Taxi environment
env = gym.make("Taxi-v3")
state_size = env.observation_space.n  # 500 discrete states
action_size = env.action_space.n  # 6 discrete actions

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration
EPISODES = 1000  # Number of episodes to train on
BATCH_SIZE = 64  # Batch size for training
MEMORY_SIZE = 10000  # Size of replay memory


# Neural Network for Deep SARSA
class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepQNetwork, self).__init__()
        # One-hot encoding for discrete state
        self.state_size = state_size
        self.action_size = action_size

        # Neural network layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        # Convert state index to one-hot vector
        if isinstance(state, int) or (
            isinstance(state, torch.Tensor) and state.dim() == 0
        ):
            x = torch.zeros(self.state_size)
            x[state] = 1.0
        else:  # Batch processing
            x = torch.zeros(len(state), self.state_size)
            for i, s in enumerate(state):
                x[i, s] = 1.0

        # Forward pass through network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Initialize neural network and optimizer
q_network = DeepQNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Initialize replay memory
memory = ReplayMemory(MEMORY_SIZE)


# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            # Choose the action with highest Q-value
            q_values = q_network(state)
            return torch.argmax(q_values).item()
    else:
        # Choose a random action
        return random.randrange(action_size)


# Training function
def train():
    epsilon = EPSILON_START
    rewards_history = []
    steps_history = []

    for episode in range(EPISODES):
        state = env.reset()[0]
        total_reward = 0
        done = False
        steps = 0

        # Select first action using epsilon-greedy
        action = select_action(state, epsilon)

        while not done:
            # Take action, observe reward and next state
            next_state, reward, done, truncated, _ = env.step(action)
            # done = done or truncated
            total_reward += reward
            steps += 1

            # Select next action using epsilon-greedy
            next_action = select_action(next_state, epsilon)

            # Store transition in memory
            memory.push(state, action, reward, next_state, next_action, done)

            # Move to the next state and action
            state = next_state
            action = next_action

            # Perform training if enough samples are available
            if len(memory) > BATCH_SIZE:
                # Sample random batch from memory
                transitions = memory.sample(BATCH_SIZE)
                (
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_next_actions,
                    batch_dones,
                ) = zip(*transitions)

                # Convert to tensors
                batch_states = torch.tensor(batch_states, dtype=torch.long)
                batch_actions = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(
                    1
                )
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float)
                batch_next_states = torch.tensor(batch_next_states, dtype=torch.long)
                batch_next_actions = torch.tensor(batch_next_actions, dtype=torch.long)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float)

                # Compute current Q values
                current_q_values = (
                    q_network(batch_states).gather(1, batch_actions).squeeze(1)
                )

                # Compute next Q values (SARSA uses the actual next action chosen)
                with torch.no_grad():
                    next_q_values = q_network(batch_next_states)
                    next_action_q_values = next_q_values.gather(
                        1, batch_next_actions.unsqueeze(1)
                    ).squeeze(1)
                    expected_q_values = batch_rewards + GAMMA * next_action_q_values * (
                        1 - batch_dones
                    )

                # Compute loss
                loss = criterion(current_q_values, expected_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Record results
        rewards_history.append(total_reward)
        steps_history.append(steps)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_steps = np.mean(steps_history[-10:])
            print(
                f"Episode {episode+1}/{EPISODES}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {epsilon:.4f}"
            )

    return rewards_history, steps_history


# Function to test the trained agent
def test_agent(num_episodes=10):
    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Select action with greedy policy (epsilon=0)
            action = select_action(state, 0.0)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        total_rewards.append(total_reward)
        total_steps.append(steps)
        print(
            f"Test Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Steps: {steps}"
        )

    print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Average Steps over {num_episodes} episodes: {np.mean(total_steps):.2f}")


# Visualize results
def plot_results(rewards, steps):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.tight_layout()
    plt.show()


# Run training
if __name__ == "__main__":
    print("Starting Deep SARSA training for Taxi-v3...")
    start_time = time.time()
    rewards, steps = train()
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # Plot training results
    plot_results(rewards, steps)

    # Test the trained agent
    print("\nTesting the trained agent...")
    test_agent()

    # Close the environment
    env.close()
