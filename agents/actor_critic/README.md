# Actor-Critic (A2C – Advantage Actor-Critic)

> Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning.* ICML. arXiv:1602.01783.

---

## Motivation

REINFORCE computes returns $G_t$ as a Monte-Carlo sum over the full episode.
This is unbiased but **high variance** — a single unlucky episode can dominate the gradient.

Actor-Critic reduces variance by subtracting a **value function baseline** $V(s_t)$ from the return:

$$A_t = G_t - V_\phi(s_t)$$

This advantage $A_t$ answers: *"was this action better or worse than average from this state?"*
It preserves the direction of the gradient while dramatically reducing its noise.

---

## Algorithm

### Loss function

$$L = \underbrace{-\mathbb{E}_t\left[A_t \cdot \log \pi_\theta(a_t|s_t)\right]}_{\text{actor loss}} + \underbrace{c_{vf} \cdot \mathbb{E}_t\left[(G_t - V_\phi(s_t))^2\right]}_{\text{critic loss}} - \underbrace{c_{ent} \cdot \mathbb{E}_t\left[\mathcal{H}[\pi_\theta(\cdot|s_t)]\right]}_{\text{entropy bonus}}$$

### Pseudocode

```
for each episode:
    τ = {s₀, a₀, r₀, …, s_T}  collected under π_θ
    G_t  = Σ_{k≥t} γ^{k-t} r_k          # Monte-Carlo returns
    A_t  = G_t − V_φ(s_t)                # advantage estimate
    A_t  = normalise(A_t)                 # zero mean, unit std

    L_actor  = −mean(A_t · log π_θ(a_t|s_t))
    L_critic =  0.5 · mean((G_t − V_φ(s_t))²)
    L_entropy= −mean(H[π_θ(·|s_t)])
    L        =  L_actor + c_vf·L_critic − c_ent·L_entropy

    θ, φ ← θ − α·∇L   (single step, shared network)
```

---

## Architecture

```
Observation (obs_dim)
       │
  Shared MLP  [hidden_dims[:-1]]   ← joint feature extraction
       │
       ├──► Actor head  Linear(hidden, act_dim) → logits / mean
       │                                              │
       │                                     Categorical / Gaussian
       │
       └──► Critic head Linear(hidden, 1) → V(s)
```

**Shared trunk** — unlike PPO (which uses separate actor/critic networks),
Actor-Critic shares all layers except the final heads.
This makes it parameter-efficient and lets the features learned for the critic
directly improve the actor's representation.

---

## Implementation decisions

| Choice | Rationale |
|--------|-----------|
| Shared trunk | Efficient; classic A2C design |
| RMSprop optimiser | Original A3C paper used RMSprop; smoother than Adam for on-policy RL |
| `vf_coef=0.5` | Standard; prevents critic from dominating |
| Advantage normalisation | Per-episode zero-mean/unit-std; critical for stability |
| `values.detach()` in actor loss | Prevents critic gradients from flowing through actor path |
| Monte-Carlo returns (not TD) | Full episode rollout; unbiased but higher variance than n-step TD |

---

## Comparison: REINFORCE → Actor-Critic → PPO

| Property | REINFORCE | Actor-Critic | PPO |
|----------|-----------|--------------|-----|
| Baseline | None | V(s) | V(s) |
| Advantage | Raw return G_t | G_t − V(s) | GAE (λ-weighted) |
| Network | Policy only | Shared actor+critic | Separate actor/critic |
| Optimiser | Adam | RMSprop | Adam |
| Sample reuse | ✗ | ✗ | ✓ (K epochs) |
| Rollout | Full episode | Full episode | Fixed T steps |
| Policy constraint | None | None | Clipping |
| Variance | High | Medium | Low |
| Stability | Fragile | Moderate | Robust |
| Complexity | Simple | Moderate | Higher |

---

## Diagnostics to monitor

| Metric | What to look for |
|--------|-----------------|
| `actor_loss` | Should decrease then stabilise; large spikes = unstable |
| `critic_loss` | Should steadily decrease as V(s) improves |
| `entropy` | Slow decay = healthy exploration; sudden drop = collapsed policy |
| `episode_return` | Main performance metric |