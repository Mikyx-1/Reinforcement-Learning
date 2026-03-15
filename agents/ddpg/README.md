# DDPG – Deep Deterministic Policy Gradient

> Lillicrap, T. P. et al. (2016). *Continuous control with deep reinforcement learning.* ICLR. arXiv:1509.02971.

---

## Motivation

DQN is powerful but limited to **discrete** action spaces — it works by computing Q(s,a) for every action and taking the argmax. In continuous spaces (Pendulum, MuJoCo, robotics), the argmax over an infinite set is intractable.

DDPG solves this by learning a **deterministic policy** μ(s) → a that directly outputs the action maximising Q:

$$\mu(s) \approx \arg\max_a Q(s, a)$$

The policy gradient for a deterministic actor is:

$$\nabla_\theta J = \mathbb{E}_s \left[ \nabla_a Q_\phi(s, a) \big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s) \right]$$

No log-probability, no stochastic sampling — just backprop through the critic into the actor.

---

## Algorithm

```
Initialise actor μ_θ, critic Q_φ, targets μ_θ', Q_φ' = copies
Initialise replay buffer B

for each step t:
    # Collect
    a_t  = clip( μ_θ(s_t) + ε_t,  a_low, a_high )    ε ~ OUNoise
    s_{t+1}, r_t, done ← env.step(a_t)
    B.push(s_t, a_t, r_t, s_{t+1}, done)

    # Sample mini-batch
    {s, a, r, s', d} ~ B

    # Update critic  (Bellman regression)
    a'  = μ_θ'(s')
    y   = r + γ · Q_φ'(s', a') · (1 − d)
    L_Q = MSE( Q_φ(s, a),  y )
    φ ← φ − α_Q ∇_φ L_Q

    # Update actor  (policy gradient through critic)
    L_μ = −mean( Q_φ(s, μ_θ(s)) )
    θ ← θ − α_μ ∇_θ L_μ

    # Soft update targets
    φ' ← τ φ + (1−τ) φ'
    θ' ← τ θ + (1−τ) θ'
```

---

## Architecture

```
Actor:
    obs → Linear(obs_dim, h1) → ReLU
        → Linear(h1, h2)      → ReLU
        → Linear(h2, act_dim) → Tanh × act_limit

Critic:
    obs → Linear(obs_dim, h1) → ReLU
        → concat(hidden, action)
        → Linear(h1+act_dim, h2) → ReLU
        → Linear(h2, 1)
```

**Why action is injected after the first layer (not at input):**
The first layer learns a nonlinear encoding of the observation. Injecting the action afterwards lets the critic learn *"given this interpretation of the state, how good is this action?"* rather than processing raw obs and action together from the start.

**Why Tanh + act_limit (not clamp):**
Clamp has zero gradient at the bounds. Tanh is smooth everywhere, so the actor receives a gradient even when trying to push the action towards the boundary.

---

## Exploration — OUNoise vs Gaussian

| Property | OUNoise | Gaussian |
|----------|---------|----------|
| Correlation | Temporally correlated (momentum) | i.i.d. per step |
| Equations | $dx = \theta(\mu - x)dt + \sigma dW$ | $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ |
| Better for | Continuous physical control, momentum tasks | Most environments |
| Decay | Natural mean-reversion | Explicit sigma decay |
| Reset | Every episode | No state to reset |

In practice, both work well. The original DDPG paper used OU noise; subsequent work (TD3, SAC) often uses simple Gaussian noise.

---

## Implementation decisions

| Choice | Rationale |
|--------|-----------|
| Separate actor/critic lr (`1e-4` / `1e-3`) | Critic needs to converge faster than actor; standard DDPG ratios |
| Soft update τ=0.005 | Very slow target movement — critical for off-policy stability |
| Freeze critic during actor update | Prevents stale critic gradients polluting the actor update; zero extra cost |
| Fan-in init for hidden, ±3e-3 for output | Original DDPG paper recommendation; keeps initial Q≈0 and μ≈0 |
| Gradient clipping `max_norm=1.0` | Smaller than DQN's 10 — continuous action gradient norms are typically smaller |
| `on_episode_end` resets OU state | Essential — without reset, OU noise accumulates across episodes |

---

## Comparison with PPO

| Property | DDPG | PPO |
|----------|------|-----|
| Action space | Continuous only | Discrete + Continuous |
| Policy type | Deterministic | Stochastic |
| Sample reuse | ✓ Replay buffer | ✓ K epochs |
| Exploration | Additive noise | Policy entropy |
| Target networks | ✓ Soft update (both actor + critic) | ✗ |
| Stability | Fragile (sensitive to lr, noise) | Robust |
| Known issues | Overestimation bias, brittle | Hyperparameter sensitivity |
| Successor | TD3 (fixes overestimation) | — |

---

## Diagnostics

| Metric | Healthy signal |
|--------|---------------|
| `critic_loss` | Should decrease in the first ~50k steps |
| `actor_loss` | Should become more negative as policy improves (we minimise -Q) |
| `mean_q` | Should rise; divergence = instability, reduce lr or increase warmup |
| `eval/mean_return` | Primary metric; expect slow start, then rapid improvement after ~100k steps |