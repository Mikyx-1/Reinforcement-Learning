"""
Graph Actor-Critic for PPO over NetworkRoutingEnv's Dict observation.

No torch_geometric dependency — message passing is done with plain tensor
ops, which is fine here because the topology is fixed for the env's whole
lifetime (edge_index is baked into the module as a buffer at construction,
not re-derived per batch):

    h_src, h_dst = h[:, src_idx], h[:, dst_idx]                  # gather
    msg          = MLP(concat(h_src, h_dst, edge_features))       # message
    agg[node]    = mean of incoming msg                           # scatter (scatter_add_ + degree)
    h            = LayerNorm(h + MLP(concat(h, agg)))              # residual update

Same act()/evaluate() interface as agents/ppo/networks.py's
CategoricalActorCritic, so PPOAgent's update loop (_update_minibatch, GAE,
clipping, save/load) is reused unchanged by GNNPPOAgent — only obs happens
to be a Dict of tensors instead of one flat tensor.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from common.utils import init_weights


def _init_mlp(net: nn.Sequential, hidden_gain: float, output_gain: float) -> None:
    linear_layers = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        gain = output_gain if i == len(linear_layers) - 1 else hidden_gain
        init_weights(layer, gain=gain)


class GraphConvLayer(nn.Module):
    """One round of message passing over a fixed, directed edge list."""

    def __init__(self, edge_index: np.ndarray, num_nodes: int, hidden_dim: int, edge_feat_dim: int):
        super().__init__()
        self.register_buffer("src_idx", torch.as_tensor(edge_index[0], dtype=torch.long))
        self.register_buffer("dst_idx", torch.as_tensor(edge_index[1], dtype=torch.long))
        num_edges = edge_index.shape[1]
        deg = torch.zeros(num_nodes)
        deg.index_add_(0, self.dst_idx, torch.ones(num_edges))
        self.register_buffer("in_degree", deg.clamp(min=1).view(1, num_nodes, 1))
        self.num_nodes = num_nodes

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        _init_mlp(self.msg_mlp, hidden_gain=2**0.5, output_gain=2**0.5)
        _init_mlp(self.update_mlp, hidden_gain=2**0.5, output_gain=2**0.5)

    def forward(self, h: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        B, N, H = h.shape
        h_src = h[:, self.src_idx, :]
        h_dst = h[:, self.dst_idx, :]
        msg = self.msg_mlp(torch.cat([h_src, h_dst, edge_features], dim=-1))

        agg = h.new_zeros(B, N, H)
        idx = self.dst_idx.view(1, -1, 1).expand(B, -1, H)
        agg.scatter_add_(1, idx, msg)
        agg = agg / self.in_degree

        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        edge_index: np.ndarray,
        num_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        init_weights(self.input_proj, gain=2**0.5)
        self.layers = nn.ModuleList(
            [GraphConvLayer(edge_index, num_nodes, hidden_dim, edge_feat_dim) for _ in range(n_layers)]
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(node_features))
        for layer in self.layers:
            h = layer(h, edge_features)
        return h


class GraphCategoricalActorCritic(nn.Module):
    """
    actor  : per-node GNN embeddings -> score each neighbor of the current
             node -> masked Categorical over `max_degree` action slots.
    critic : per-node GNN embeddings -> mean-pool + current-node embedding -> V(s).

    Separate actor/critic encoders (same rationale as PPOAgent's plain MLP
    version: K update epochs per rollout, shared trunks would let the value
    and policy objectives fight each other across epochs).
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        neighbor_table: np.ndarray,
        node_feat_dim: int = 3,
        edge_feat_dim: int = 3,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        num_nodes, max_degree = neighbor_table.shape
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim
        self.register_buffer("neighbor_table", torch.as_tensor(neighbor_table, dtype=torch.long))

        self.actor_encoder = GNNEncoder(edge_index, num_nodes, node_feat_dim, edge_feat_dim, hidden_dim, n_layers)
        self.critic_encoder = GNNEncoder(edge_index, num_nodes, node_feat_dim, edge_feat_dim, hidden_dim, n_layers)

        self.action_head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        _init_mlp(self.action_head, hidden_gain=2**0.5, output_gain=0.01)
        _init_mlp(self.value_head, hidden_gain=2**0.5, output_gain=1.0)

    def _dist_and_value(self, obs: dict[str, torch.Tensor]):
        node_features = obs["node_features"]
        edge_features = obs["edge_features"]
        action_mask = obs["action_mask"]
        B, H = node_features.shape[0], self.hidden_dim

        current_idx = node_features[..., 1].argmax(dim=1)  # is_current flag
        batch_arange = torch.arange(B, device=node_features.device)

        h_actor = self.actor_encoder(node_features, edge_features)
        h_cur = h_actor[batch_arange, current_idx]
        neighbor_ids = self.neighbor_table[current_idx]  # (B, max_degree)
        gather_idx = neighbor_ids.unsqueeze(-1).expand(-1, -1, H)
        neighbor_embeds = torch.gather(h_actor, 1, gather_idx)
        h_cur_expand = h_cur.unsqueeze(1).expand(-1, self.max_degree, -1)
        logits = self.action_head(torch.cat([h_cur_expand, neighbor_embeds], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        h_critic = self.critic_encoder(node_features, edge_features)
        pooled = h_critic.mean(dim=1)
        h_cur_c = h_critic[batch_arange, current_idx]
        value = self.value_head(torch.cat([pooled, h_cur_c], dim=-1)).squeeze(-1)

        return Categorical(logits=logits), value

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        dist, value = self._dist_and_value(obs)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        dist, value = self._dist_and_value(obs)
        log_prob = dist.log_prob(actions.long().squeeze(-1))
        return log_prob, dist.entropy(), value
