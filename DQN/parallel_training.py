import copy
import math
import random
import threading
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import List, Union

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn, optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity: int = 2000) -> None:
        self.memory = deque([], maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, *args) -> None:
        """Save a transition in a thread-safe manner"""
        with self.lock:
            self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        with self.lock:
            return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        with self.lock:
            return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
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
        env: str = "CartPole-v1",
        memory_size: int = 10000,
        device: Union[torch.device, str] = "cpu",
        batch_size: int = 128,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
    ) -> None:
        self.env_name = env
        self.policy_agent = agent.to(device)
        self.device = device
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.steps_done = 0

    def init_training_phase(self) -> None:
        self.target_agent = copy.deepcopy(self.policy_agent).to(self.device)
        self.target_agent.eval()
        self.policy_agent.train()
        self.optimizer = optim.AdamW(
            self.policy_agent.parameters(), lr=self.lr, amsgrad=True
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer.zero_grad()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 0.001
        # print(f"steps_done: {self.steps_done}, eps_threshold: {eps_threshold}")
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_agent(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.policy_agent.layer3.out_features)]],
                device=self.device,
            )

    def accumulate_gradients(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        nonfinal_mask = torch.tensor(
            [s is not None for s in batch.next_state], device=self.device
        )
        nonfinal_next = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state).float().to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        state_action = self.policy_agent(state_batch).gather(1, action_batch)
        next_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_values[nonfinal_mask] = self.target_agent(nonfinal_next).max(1)[0]
        expected = reward_batch + self.gamma * next_values
        loss = self.loss_fn(state_action, expected.unsqueeze(1))
        loss.backward()

    def update_parameters(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.policy_agent.parameters(), 40)
        self.optimizer.step()
        # soft update
        for p, t in zip(self.policy_agent.parameters(), self.target_agent.parameters()):
            t.data.mul_(1 - self.tau)
            t.data.add_(p.data * self.tau)
        self.optimizer.zero_grad()

    def rollout(self) -> float:
        env = gym.make(self.env_name)
        state, _ = env.reset()
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        total_reward = 0
        for _ in count():
            action = self.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward_t = torch.tensor([reward], device=self.device)
            done = terminated or truncated
            next_state = (
                None if done else torch.from_numpy(obs).unsqueeze(0).to(self.device)
            )
            self.memory.push(state, action, next_state, reward_t)
            self.accumulate_gradients()
            state = next_state
            if done:
                break
        env.close()
        return total_reward

    def train(self, num_episodes: int = 600, num_threads: int = 4) -> None:
        self.init_training_phase()
        for ep in range(num_episodes):
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(self.rollout) for _ in range(num_threads)]
                rewards = [f.result() for f in futures]
            self.update_parameters()
            avg_r = sum(rewards) / len(rewards)
            print(f"Episode batch {ep}, Avg Reward: {avg_r:.2f}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    agent = DQN(n_obs, n_act).to(device)
    trainer = DQNTrainer(
        agent, env="CartPole-v1", memory_size=10000, device=device, batch_size=512
    )
    trainer.train(num_episodes=100000, num_threads=4)
