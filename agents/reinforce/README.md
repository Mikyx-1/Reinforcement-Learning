# REINFORCE – Monte-Carlo Policy Gradient

> Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 8, 229–256.

---

## Algorithm

REINFORCE is the simplest policy gradient algorithm. It directly optimises the expected return by following the gradient of the log-policy weighted by the observed return.

### Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

where $G_t = \sum_{k \geq t} \gamma^{k-t} r_k$ is the discounted return from step $t$.

### Pseudocode

```
for each episode:
    τ = rollout under π_θ
    for each t in τ:
        G_t = Σ_{k≥t} γ^{k-t} r_k
    normalise G_t  →  (G_t - mean) / (std + ε)
    loss = -mean( G_t · log π_θ(a_t|s_t) )
    θ  ←  θ - α · ∇_θ loss
```

### Entropy regularisation (optional)

Adding an entropy bonus encourages exploration by preventing premature convergence to deterministic policies:

$$\mathcal{L} = -\mathbb{E}[G_t \log \pi_\theta(a_t|s_t)] - \beta \cdot \mathcal{H}[\pi_\theta]$$

---

## Implementation notes

| Design choice | Rationale |
|---------------|-----------|
| **Normalise returns** `(G - mean) / std` | Reduces variance significantly; standard practice |
| **Recompute log probs** from current policy | Ensures correct gradient even if rollout is slightly stale |
| **Orthogonal init, gain=0.01** on output layer | Keeps initial action distribution near-uniform; faster convergence |
| **Gradient clipping** `max_norm=0.5` | Guards against rare large updates on long episodes |
| **Separate CategoricalPolicy / GaussianPolicy** | Clean separation of discrete vs continuous action spaces |

---

## Known limitations

- **High variance** – each episode is a single Monte-Carlo sample. PPO and A2C address this with a value baseline.
- **On-policy** – samples cannot be reused; sample efficiency is low compared to off-policy methods.
- **No credit assignment** – all actions in the episode receive the full return $G_0$, not just the return from their timestep. The $G_t$ formulation partially fixes this but does not subtract a baseline.

**Next step:** Add a value function baseline → Actor-Critic → PPO.

---

## Results on CartPole-v1

| Metric | Value |
|--------|-------|
| Solve threshold | mean return ≥ 475 |
| Typical solve episode | ~300–600 |
| Hyperparameters | See `configs/reinforce_cartpole.yaml` |