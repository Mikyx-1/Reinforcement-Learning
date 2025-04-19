import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count
from typing import Union
import torch
from torch import nn, optim
import torch.nn.functional as F
import copy

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity: int = 2000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNTrainer(object):
    def __init__(
        self,
        agent: nn.Module,
        env: str = "LunarLander-v3",
        memory_size: int = 1000,
        device: Union[torch.device, str] = "cpu",
        batch_size: int = 128,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
    ):

        self.agent = agent
        self.memory = self.init_memory(memory_size)
        self.device = device
        self.env = gym.make(env)

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.steps_done = 0
        self.n_actions = self.env.action_space.n

    def init_memory(self, memory_size: int = 1000) -> object:
        memory = ReplayMemory(memory_size)
        return memory

    def init_training_phase(self) -> None:
        self.running_agent = copy.deepcopy(self.agent)
        self.running_agent.train()
        self.agent.train()

        self.optimiser = optim.AdamW(
            self.running_agent.parameters(), lr=self.lr, amsgrad=True
        )
        self.loss_fn = nn.SmoothL1Loss()

    def random_act(self) -> torch.Tensor:
        return torch.tensor(
            [[self.env.action_space.sample()]], device=self.device, dtype=torch.long
        )

    def deterministic_act(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.agent(state).max(1)[1].view(1, 1)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            return self.deterministic_act(state)
        else:
            return self.random_act()

    def train_agent(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.agent(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.running_agent(
                non_final_next_states
            ).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.loss_fn(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        self.optimiser.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 40)
        self.optimiser.step()

        # Soft update of the target network's weights
        agent_state_dict = self.agent.state_dict()
        running_agent_state_dict = self.running_agent.state_dict()

        for key in running_agent_state_dict.keys():
            agent_state_dict[key] = (1 - self.tau) * agent_state_dict[
                key
            ] + self.tau * running_agent_state_dict[key]

        self.agent.load_state_dict(agent_state_dict)

    def rollout(self) -> int:

        state, _ = self.env.reset()

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        total_rewards = 0
        for t in count():
            action = self.select_action(state)
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            total_rewards += reward
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(observation).unsqueeze(0).to(self.device)

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.train_agent()

            if done:
                break

        return total_rewards, t

    def train(self, num_episodes: int = 600) -> None:
        self.init_training_phase()
        for i_episode in range(num_episodes):
            total_rewards, eps_length = self.rollout()
            print(f"Episode {i_episode}, Episode length: {eps_length} Total rewards: {total_rewards}")
        self.env.close()


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    agent = DQN(n_observations, n_actions).to(device)
    trainer = DQNTrainer(agent, device=device)
    trainer.train()
