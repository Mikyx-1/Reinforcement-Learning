import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import numpy as np
import copy
import math


class BetaActor(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, net_width: int = 128):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode


class Critic(nn.Module):
	def __init__(self, state_dim: int ,net_width: int = 128):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v


class PPO_agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
        self.actor = BetaActor(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic = Critic(state_dim=self.state_dim)
        
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
		# Build Trajectory holder
        self.s_holder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
        self.a_holder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
        self.r_holder = np.zeros((self.T_horizon, 1),dtype=np.float32)
        self.s_next_holder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
        self.logprob_a_holder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
        self.done_holder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
        self.dw_holder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
        

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if deterministic:
                # only used when evaluate the policy. Making the performane more stable
                a = self.actor.deterministic_act(state)
                return a.cpu().numpy()[0], None
            
            else:
                # only used when interact with the env
                dist = self.actor.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy()[0], logprob_a
            
            
    def train(self):
        self.entropy_coef *= self.entropy_coef_decay
        
        '''Prepare Pytorch data from Numpy data'''
        s = torch.from_numpy(self.s_holder).to(self.device)
        a = torch.from_numpy(self.a_holder).to(self.device)
        r = torch.from_numpy(self.r_holder).to(self.device)
        s_next = torch.from_numpy(self.s_next_holder).to(self.device)
        logprob_a = torch.from_numpy(self.logprob_a_holder).to(self.device)
        done = torch.from_numpy(self.done_holder).to(self.device)
        dw = torch.from_numpy(self.dw_holder).to(self.device)
        
        
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)
            
            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            
            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
                adv.append(advantage)

            adv.reverse()
            adv = copy.deepcopy((adv[::-1]))
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)
            
            """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
            a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
            c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
            
            for i in range(self.K_epochs):
                # Shuffle the trajectory, Good for training
                perm = np.arange(s.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(self.device)
                s = s[perm].clone()
                a = a[perm].clone()
                td_target = td_target[perm].clone()
                adv = adv[perm].clone()
                logprob_a = logprob_a[perm].clone()
                
                
                for i in range(a_optim_iter_num):
                    index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
                    distribution = self.actor.get_dist(s[index]) 
                    dist_entropy = distribution.entropy().sum(1, keepdim=True)
                    logprob_a_now = distribution.log_prob(a[index])
                    ratio = torch.exp(logprob_a_now.sum(1, keepdim=True) - logprob_a[index].sum(1, keepdim=True))
                    
                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1-self.clip_rate, 1+self.clip_rate)*adv[index]
                    a_loss = -torch.min(surr1, surr2) - self.entropy_coef*dist_entropy
                    
                    self.actor_optimiser.zero_grad()
                    a_loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                    self.actor_optimiser.step()
                    
                    
            for i in range(c_optim_iter_num):
                index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic[s[index]] - td_target).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if "weight" in name:
                        c_loss += param.pow(2).sum() *self.l2_reg
                        
                self.critic_optimiser.zero_grad()
                c_loss.backward()
                self.critic_optimiser.step()
                
                
    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_holder[idx] = s
        self.a_holder[idx] = a
        self.r_holder[idx] = r
        self.s_next_holder[idx] = s_next
        self.logprob_a_holder[idx] = logprob_a
        self.done_holder[idx] = done
        self.dw[idx] = dw

if __name__ == "__main__":
    # Define hyperparameters (example values, adjust as needed)
    params = {
        'state_dim': 8,              # for example, state dimension
        'action_dim': 2,             # for example, action dimension in [0,1]
        'net_width': 64,
        'device': torch.device("cpu"),
        'T_horizon': 2048,
        'actor_lr': 2e-4,
        'critic_lr': 2e-4,
        'gamma': 0.99,
        'lambd': 0.95,
        'a_optim_batch_size': 64,
        'c_optim_batch_size': 64,
        'K_epochs': 10,
        'clip_rate': 0.2,
        'entropy_coef': 1e-3,
        'entropy_coef_decay': 0.99,
        'l2_reg': 1e-3
    }
    
    # Create agent instance.
    agent = PPO_agent(**params)
    
    # (Your training loop would interact with the environment here, call select_action,
    # store transitions via put_data, and call agent.train() once the trajectory is filled.)
    print("PPO agent for continuous control with Beta distribution is ready.")
