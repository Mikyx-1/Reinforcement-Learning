import gymnasium as gym
import torch
from torch import nn, optim
import numpy as np
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
from itertools import count
import math

device = "cpu"


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

            
class Policy(nn.Module):
    def __init__(self, n_observations: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_observations, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        # out = self.tanh(self.l3(x)) * 2.
        out = torch.clamp(self.l3(x), -2, 2)
        return out
    
class QNetwork(nn.Module):
    def __init__(self, n_observations: torch.Tensor) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_observations + 1, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)
    


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args) -> None:
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

## Environment    
env = gym.make("Pendulum-v1")
obs, info = env.reset()
n_observations = len(obs)

## Policy Net
PolicyNet = Policy(n_observations)
PolicyTarget = Policy(n_observations)
PolicyTarget.load_state_dict(PolicyNet.state_dict())

## Q Net
QNet = QNetwork(n_observations)
QNetTarget = QNetwork(n_observations)
QNetTarget.load_state_dict(QNet.state_dict())



## Hyperparams
LR = 1e-3
NUM_EPOCHS = 750
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 10000

## Auxiliary vars

memory = ReplayBuffer(10000)
steps_done = 0
noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2)*np.ones(1))
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

policy_optimiser = optim.Adam(PolicyNet.parameters(), lr=LR, amsgrad=True) 
qnet_optimiser = optim.Adam(QNet.parameters(), lr=LR, amsgrad=True)


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1.*steps_done/EPS_DECAY)
    sample = random.random()
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return PolicyNet(state).view(1).float()
    return torch.tensor(env.action_space.sample(), dtype=torch.float32)


def optimise_model():
    if len(memory) < BATCH_SIZE:
        return

    batch = Transition(*zip(*memory.sample(BATCH_SIZE)))
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    

    with torch.no_grad():
        target_actions = PolicyTarget(next_state_batch)
        y = reward_batch.unsqueeze(1) + GAMMA*QNetTarget(next_state_batch, target_actions)

    critic_value = QNet(state_batch, action_batch)
    critic_loss = nn.MSELoss()(critic_value, y)
    QNet.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_value_(QNet.parameters(), 80)
    qnet_optimiser.step()

    actions = PolicyNet(state_batch)
    policy_loss = -torch.mean(QNet(state_batch, actions))
    PolicyNet.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_value_(PolicyNet.parameters(), 80)
    policy_optimiser.step()

    

env = gym.make("Pendulum-v1")
for i_episode in range(NUM_EPOCHS):
    rewards = []
    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32).view(1, 3)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.numpy())
        rewards.append(reward)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = terminated or truncated
        

        next_state = torch.tensor(observation, dtype=torch.float32).view(1, 3)
            
        memory.push(state, action, reward, next_state)
        state = next_state
        
        optimise_model()

        policy_state_dict = PolicyNet.state_dict()
        policy_target_state_dict = PolicyTarget.state_dict()
        QNet_state_dict = QNet.state_dict()
        QNetTarget_state_dict = QNetTarget.state_dict()
        
        for key in policy_state_dict:
            policy_target_state_dict[key] = TAU*policy_state_dict[key] + (1-TAU)*policy_target_state_dict[key]
        PolicyTarget.load_state_dict(policy_target_state_dict)
        
        for key in QNet_state_dict:
            QNetTarget_state_dict[key] = TAU*QNet_state_dict[key] + (1-TAU)*QNetTarget_state_dict[key]
        QNetTarget.load_state_dict(QNetTarget_state_dict)
        
        if done:
            break
    print(f"Episode: {i_episode}        Performance: {sum(rewards)}")

torch.save(PolicyNet.state_dict(), "PolicyNet_PendulumV2.pt")