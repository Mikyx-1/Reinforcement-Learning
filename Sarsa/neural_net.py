import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.5
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
ALPHA = 0.01
GAMMA = 0.99
BATCH_SIZE = 64
env = gym.make("Taxi-v3")

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "next_action", "done")
)


class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory.clear()

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)


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
            x = torch.zeros(self.state_size, device=DEVICE)
            x[state] = 1.0
        else:  # Batch processing
            x = torch.zeros(len(state), self.state_size, device=DEVICE)
            for i, s in enumerate(state):
                x[i, s] = 1.0

        # Forward pass through network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


policy = DeepQNetwork(env.observation_space.n, env.action_space.n).to(DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=ALPHA)
memory = ReplayMemory()


# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            # Choose the action with highest Q-value
            q_values = policy(state)
            return torch.argmax(q_values).item()
    else:
        # Choose a random action
        return random.randrange(env.action_space.n)


def rollout():
    global EPSILON
    state, _ = env.reset()
    done = False

    state = torch.as_tensor(state, device=DEVICE, dtype=torch.long).view(-1, 1)
    action = select_action(state, EPSILON)

    total_rewards = 0

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.as_tensor(next_state, device=DEVICE, dtype=torch.long).view(
            -1, 1
        )
        done = terminated or truncated

        next_action = select_action(next_state, EPSILON)

        memory.push(state, action, reward, next_state, next_action, done)

        state = next_state
        action = next_action
        total_rewards += reward

        train_policy(memory)

    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    return total_rewards


def train_policy(memory):
    if len(memory) < BATCH_SIZE:
        return

    transitions = Transition(*zip(*memory.sample(BATCH_SIZE)))
    state_batch = torch.cat(transitions.state).to(DEVICE)
    action_batch = torch.tensor(transitions.action, device=DEVICE).view(-1, 1)
    reward_batch = torch.tensor(
        transitions.reward, device=DEVICE, dtype=torch.float32
    ).view(-1, 1)
    next_state_batch = torch.cat(transitions.next_state).view(-1, 1).to(DEVICE)
    next_action_batch = torch.tensor(transitions.next_action, device=DEVICE).view(-1, 1)
    done_batch = torch.tensor(transitions.done, device=DEVICE, dtype=torch.float).view(
        -1, 1
    )

    state_action_values = policy(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_state_action_values = policy(next_state_batch).gather(1, next_action_batch)

        expected_state_action_values = (
            reward_batch + GAMMA * next_state_action_values * (1 - done_batch)
        )

    loss = nn.functional.huber_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 100000
for episode in range(num_episodes):
    total_rewards = rollout()

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1} completed. Total_rewards: {total_rewards}")
