# PPO – Proximal Policy Optimisation

> Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
> *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

---

## Motivation

Actor-Critic reduces variance with a value baseline, but it offers **no constraint on how far the policy moves in a single update**. A bad gradient step can collapse the policy, and there is no sample reuse — each episode is discarded after one update.

PPO solves both problems:
- **Clipping** prevents catastrophic updates by bounding the policy change per step.
- **Fixed-T rollouts + K epochs** reuse the same data multiple times safely.

---

## Algorithm

### 1. Generalised Advantage Estimation (GAE)

$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}, \qquad
\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$$

- $\lambda = 0$ → pure TD residual (low variance, high bias)
- $\lambda = 1$ → Monte-Carlo (high variance, low bias)
- $\lambda = 0.95$ is the standard working point

### 2. Clipped Surrogate Objective

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(
  r_t(\theta)\,\hat{A}_t,\;
  \operatorname{clip}(r_t(\theta), 1{-}\varepsilon, 1{+}\varepsilon)\,\hat{A}_t
\right)\right], \quad
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

The clip removes gradient incentive to push the ratio outside $[1-\varepsilon, 1+\varepsilon]$.

### 3. Value Loss (clipped)

$$L^{\text{VF}} = \frac{1}{2}\,\mathbb{E}_t\!\left[\max\!\left(
  (V_\theta(s_t) - R_t)^2,\;
  (\operatorname{clip}(V_\theta, V_{\text{old}}\pm\varepsilon) - R_t)^2
\right)\right]$$

### 4. Combined Objective

$$L = -L^{\text{CLIP}} + c_{vf}\cdot L^{\text{VF}} - c_{ent}\cdot\mathcal{H}[\pi_\theta]$$

### 5. Pseudocode

```
initialise π_θ (actor), V_φ (critic)

for each iteration:
    collect T env steps under π_θ_old
    bootstrap: last_value = V(s_T) if not done else 0
    compute GAE advantages  Â_t  and returns  R_t = Â_t + V(s_t)
    normalise Â_t → zero mean, unit std (over the rollout)

    for epoch = 1..K:
        for each mini-batch B of size M (shuffled):
            r_t   = exp(log π_θ(a|s) − log π_old(a|s))
            L_CLIP = −mean(min(r·Â, clip(r, 1−ε, 1+ε)·Â))
            L_VF   = 0.5·mean((V_θ − R)²)
            L_H    = −mean(H[π_θ])
            L      = L_CLIP + c_vf·L_VF + c_ent·L_H
            θ, φ  ← Adam step on ∇L  (grad clip max_norm)
        if approx_KL > 1.5·target_KL: break early
```

---

## Architecture

```
Observation (obs_dim)
     │                           │
     ▼                           ▼
Actor MLP  [hidden_dims]    Critic MLP  [hidden_dims]
     │                           │
logits / mean               V(s)  (scalar)
     │
Categorical / Gaussian
```

**Separate networks** — unlike the shared-trunk design in `actor_critic/`, PPO's K-epoch updates create conflicting gradients between the policy objective and the value objective. Separate parameters decouple them completely.

**State-independent log_std** for continuous actions — one learnable scalar per action dimension, not a function of the observation.

---

## Implementation decisions

| Choice | Rationale |
|--------|-----------|
| Separate actor/critic | Decouples gradient flows during K-epoch updates |
| Fixed-T rollouts (not episode-aligned) | Enables consistent batch sizes; multi-episode rollouts supported |
| GAE (not raw return) | Lower variance than Monte-Carlo; better credit assignment than TD(0) |
| Advantage normalisation per rollout | Desensitises training to reward scale across environments |
| Value clipping | Prevents critic from moving too far from old estimate (conservative updates) |
| `target_kl` early stopping | Exits epoch loop if KL exceeds threshold; prevents overfitting the rollout |
| Adam with `eps=1e-5` | Smaller epsilon than default (1e-8) improves stability in PPO |
| `max_grad_norm=0.5` | Clips explosive gradients near episode boundaries |

---

## Diagnostics

Watch these in TensorBoard to understand what is happening:

| Metric | Healthy range | Signal |
|--------|--------------|--------|
| `approx_kl` | 0.005–0.02 | Policy update magnitude; too large → reduce lr or clip_eps |
| `clip_fraction` | 0.05–0.20 | Fraction of ratios clipped; consistently high → learning rate too large |
| `explained_variance` | → 1.0 | How well V(s) predicts returns; low = critic not learning |
| `entropy` | Slow decrease | Exploration; sudden drop → try higher ent_coef |
| `value_loss` | Monotone decrease | Critic fit quality |

---

## Comparison with Actor-Critic

| Property | Actor-Critic | PPO |
|----------|-------------|-----|
| Networks | Shared trunk | Separate actor/critic |
| Advantage | A_t = R_t − V(s) | GAE λ-return |
| Rollout | Full episode | Fixed T steps |
| Sample reuse | ✗ (1 update/episode) | ✓ (K epochs) |
| Policy constraint | None | Clipping ε |
| Optimiser | RMSprop | Adam |
| Diagnostics | 4 scalars | 7 scalars |
| Stability | Moderate | High |
| Sample efficiency | Low | High |