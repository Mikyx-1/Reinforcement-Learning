# DQN – Deep Q-Network

> Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533.

With standard improvements:
- **Double DQN** — van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
- **Dueling Networks** — Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML.

---

## Motivation

Tabular Q-learning is exact but does not scale. DQN replaces the Q-table with a neural network and introduces two stabilisation tricks that make gradient-based Q-learning tractable:

1. **Experience Replay** — break temporal correlations by storing transitions in a circular buffer and sampling random mini-batches.
2. **Target Network** — decouple the target from the online network to prevent oscillating update targets; sync periodically via hard copy.

---

## Algorithm

### Q-learning update (Bellman equation)

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \underbrace{r_t + \gamma \max_{a'} Q_{\text{target}}(s_{t+1}, a')}_{\text{TD target } y_t} - Q(s_t, a_t) \right]$$

As a supervised regression loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{B}} \left[ \text{Huber}(Q_\theta(s, a),\; y) \right]$$

### Double DQN — decoupled action selection and evaluation

Vanilla DQN is biased towards overestimation because the same network both selects and evaluates the greedy next action. Double DQN fixes this:

$$a^* = \arg\max_{a'} Q_{\text{online}}(s', a') \quad \text{(online net selects)}$$
$$y = r + \gamma \cdot Q_{\text{target}}(s', a^*) \cdot (1 - d) \quad \text{(target net evaluates)}$$

### Dueling Architecture

Splits the Q-network into two streams that share a feature trunk:

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

- $V(s)$: scalar state-value — improves value estimates for states where action choice doesn't matter
- $A(s,a)$: advantage — captures relative action preferences

### Pseudocode

```
initialise Q_online, Q_target = copy(Q_online)
initialise replay buffer B of capacity N
ε = ε_start

for each step t:
    a_t = argmax Q_online(s_t)  w.p. (1−ε),  else random
    s_{t+1}, r_t, done ← env.step(a_t)
    B.push(s_t, a_t, r_t, s_{t+1}, done)
    ε ← linear_decay(t)

    if |B| ≥ batch_size:
        sample mini-batch from B
        a* = argmax_a Q_online(s')           # Double DQN
        y  = r + γ · Q_target(s', a*) · (1−d)
        L  = Huber(Q_online(s, a), y)
        gradient step on θ

    if t % target_update_freq == 0:
        Q_target ← Q_online               # hard copy
```

---

## Architecture

```
Vanilla QNetwork:
    obs → MLP [hidden_dims] ReLU → Q(s, a₀), …, Q(s, aₙ)

DuelingQNetwork:
    obs → shared trunk → ┬→ value head    → V(s)          (scalar)
                         └→ advantage head → A(s, a₀…aₙ)
                         recombine → Q(s,a) = V(s) + A(s,a) − mean(A)
```

---

## Implementation decisions

| Choice | Rationale |
|--------|-----------|
| Huber loss (SmoothL1) | Less sensitive to outlier TD errors than MSE; more stable training |
| Hard target update (not Polyak) | Simpler; Polyak averaging is more suited to continuous-action off-policy |
| `eps_decay_steps` linear schedule | Standard; easy to reason about exploration horizon |
| Gradient clipping `max_norm=10` | Prevents explosive gradients from noisy early replay samples |
| `use_double=True` by default | Almost always better than vanilla; no extra cost |
| `use_dueling=False` by default | Useful in large action spaces; overkill for CartPole/LunarLander |

---

## Comparison with SARSA

| Property | DQN | SARSA |
|----------|-----|-------|
| Policy type | Off-policy | On-policy |
| Function approx. | Deep neural network | Linear or MLP |
| Exploration | ε-greedy (decoupled from update) | ε-greedy (same policy trains and acts) |
| Memory | Experience replay buffer | None |
| Target network | Yes | No |
| Stability | High (replay + target) | Moderate |
| Convergence guarantee | Approximate | Approximate (linear FA) |
| Best for | Complex obs, large state spaces | Simple envs, tabular insight |

---

## Diagnostics

| Metric | Healthy signal |
|--------|---------------|
| `loss` | Should decrease over first ~50k steps; small oscillations are normal |
| `mean_q` | Should rise as policy improves; divergence = instability |
| `epsilon` | Monotone decrease to `eps_end` — verify it is decaying |
| `eval/mean_return` | Primary metric; should improve steadily after warmup |