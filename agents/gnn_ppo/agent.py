"""
GNN-PPO — PPO-Clip with a graph-neural-network policy, for environments whose
observation is a Dict of (node_features, edge_index, edge_features,
action_mask) instead of a flat vector — see envs/network_routing.py.

Subclasses PPOAgent purely to reuse its already-correct, tested PPO-Clip
math (_update_minibatch, GAE bootstrapping, K-epoch update loop, save/load):
that logic only ever touches `self.ac` and `self.buffer` through their
act()/evaluate()/push()/get_batches() interfaces, never assumes obs is a flat
tensor, so it works unchanged once those two are swapped for graph-aware
versions. PPOAgent.__init__ is intentionally NOT called (obs_dim there is a
scalar, meaningless for a Dict observation space) — everything it would have
set up is set up here instead.

Only select_action() and finish_rollout() need overriding, since PPOAgent's
versions call self.to_tensor(obs) which assumes obs is array-like, not a
Dict of arrays.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from agents.base_agent import BaseAgent
from agents.gnn_ppo.buffer import GraphPPORolloutBuffer
from agents.gnn_ppo.networks import GraphCategoricalActorCritic
from agents.ppo.agent import PPOAgent


def _build_neighbor_table(env: gym.Env) -> np.ndarray:
    """(num_nodes, max_degree) int64 — env.neighbors padded with 0 (masked out by action_mask)."""
    unwrapped = env.unwrapped
    num_nodes = env.observation_space["node_features"].shape[0]
    table = np.zeros((num_nodes, unwrapped.max_degree), dtype=np.int64)
    for node, neighbor_ids in unwrapped.neighbors.items():
        table[node, : len(neighbor_ids)] = neighbor_ids
    return table


class GNNPPOAgent(PPOAgent):
    """
    Args mirror PPOAgent's, minus `hidden_dims` (replaced by `hidden_dim` +
    `n_layers` for the GNN encoder). See envs/network_routing.py for what
    `env` needs to expose (edge_index, neighbors, max_degree).
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        clip_value_loss: bool = True,
        target_kl: float | None = 0.015,
        device: str = "cpu",
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict), (
            "GNNPPOAgent requires a graph Dict observation space (node_features/"
            "edge_index/edge_features/action_mask) — see NetworkRoutingEnv."
        )
        # BaseAgent.__init__ directly: PPOAgent.__init__ expects a flat obs_dim,
        # which doesn't mean anything for a Dict observation space.
        BaseAgent.__init__(self, obs_dim=0, act_dim=env.action_space.n, device=device)
        self.discrete = True

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.target_kl = target_kl

        num_nodes, node_feat_dim = env.observation_space["node_features"].shape
        num_edges, edge_feat_dim = env.observation_space["edge_features"].shape
        neighbor_table = _build_neighbor_table(env)

        self.ac = GraphCategoricalActorCritic(
            edge_index=env.unwrapped.edge_index,
            neighbor_table=neighbor_table,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        self.buffer = GraphPPORolloutBuffer(
            rollout_steps=rollout_steps,
            num_nodes=num_nodes,
            num_edges=num_edges,
            max_degree=env.action_space.n,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            gamma=gamma,
            lam=lam,
            device=device,
        )

        self._last_value: float = 0.0

    # ------------------------------------------------------------------
    # Overrides — everything else (collect_step, update, save, load) is
    # inherited unchanged from PPOAgent.
    # ------------------------------------------------------------------

    def _obs_to_tensors(self, obs: dict) -> dict[str, torch.Tensor]:
        return {
            "node_features": self.to_tensor(obs["node_features"]).unsqueeze(0),
            "edge_features": self.to_tensor(obs["edge_features"]).unsqueeze(0),
            "action_mask": self.to_tensor(obs["action_mask"]).unsqueeze(0),
        }

    def select_action(self, obs: dict, deterministic: bool = False) -> tuple[np.ndarray, float]:
        obs_t = self._obs_to_tensors(obs)
        with torch.no_grad():
            action, log_prob, _, value = self.ac.act(obs_t, deterministic=deterministic)
        self._last_value = float(value.cpu().item())
        action_np = action.cpu().numpy().squeeze()
        log_prob_np = float(log_prob.cpu().item())
        return action_np, log_prob_np

    def finish_rollout(self, last_obs: dict, last_done: bool) -> None:
        with torch.no_grad():
            last_obs_t = self._obs_to_tensors(last_obs)
            _, _, _, last_value = self.ac.act(last_obs_t)
        bootstrap = 0.0 if last_done else float(last_value.cpu().item())
        self.buffer.compute_gae(last_value=bootstrap)
