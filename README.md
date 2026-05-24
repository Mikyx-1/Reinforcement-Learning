# Reinforcement Learning Zoo

![repo_logo](assets/rl_logo.svg)

Clean, well-documented implementations of core RL algorithms, built to demonstrate scientific rigour: reproducible experiments, structured evaluation, and clear learning curves.



---

## Algorithms

| Algorithm | Type | Action Space | Status |
|-----------|------|-------------|--------|
| REINFORCE | On-policy PG | Discrete / Continuous | ✅ Done |
| Actor-Critic | On-policy PG w/ baseline | Discrete / Continuous | ✅ Done |
| PPO | On-policy actor-critic | Discrete / Continuous | ✅ Done |
| DQN | Off-policy value | Discrete | ✅ Done |
| SARSA (semi-gradient) | On-policy value | Discrete | ✅ Done |
| DDPG | Off-policy actor-critic | Continuous | ✅ Done |
| TD3 | Off-policy actor-critic | Continuous | 🔜 |
| SAC | Off-policy actor-critic | Continuous | 🔜 |

---

## Repo structure

```
Reinforcement-Learning/
│
├── configs/                  # YAML hyperparameter files (one per experiment)
├── envs/
│   └── wrappers.py           # make_env factory + RecordEpisodeStats / NormalizeObs / ClipReward
│
├── agents/
│   ├── base_agent.py         # Abstract interface: select_action / update / save / load
│   ├── reinforce/            # ReinforceAgent + Categorical/Gaussian policies
│   ├── actor_critic/         # ActorCriticAgent (MC returns + learned baseline)
│   ├── ppo/                  # PPOAgent + its own fixed-T rollout buffer
│   ├── dqn/                  # DQNAgent (target net, soft updates, ε-greedy)
│   ├── sarsa/                # SarsaAgent (semi-gradient, on-policy)
│   └── ddpg/                 # DDPGAgent + actor/critic nets + exploration noise
│       └── (each agent dir has agent.py, networks.py, and an algorithm README)
│
├── common/
│   ├── replay_buffer.py      # ReplayBuffer (off-policy) + RolloutBuffer (on-policy MC returns)
│   ├── logger.py             # W&B + CSV logger
│   ├── schedulers.py         # LinearSchedule, ExponentialSchedule (ε decay)
│   └── utils.py              # build_mlp, soft_update, set_seed, load_config, explained_variance
│
├── training/
│   └── trainer.py            # Four loops: on-policy / actor-critic / ppo / off-policy / sarsa
│
├── evaluation/
│   └── evaluator.py          # evaluate_agent(), Evaluator (CSV export)
│
├── scripts/
│   ├── train.py              # CLI: python scripts/train.py --config ...
│   ├── evaluate.py           # CLI: evaluate a saved checkpoint
│   └── record_video.py       # CLI: record an MP4 rollout of a trained policy
│
├── results/                  # Checkpoints, CSV metrics, W&B runs (git-ignored)
└── tests/
    └── test_core.py          # Unit tests for buffers, schedulers, save/load
```

---

## Quick start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Train — algorithm is inferred from the config filename's prefix
python scripts/train.py --config configs/reinforce_cartpole.yaml
python scripts/train.py --config configs/dqn_cartpole.yaml --device cuda --seed 123
python scripts/train.py --config configs/ppo_lunarlander.yaml
python scripts/train.py --config configs/ddpg_pendulum.yaml

# Evaluate a checkpoint
python scripts/evaluate.py \
    --config configs/reinforce_cartpole.yaml \
    --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt \
    --n_episodes 20

# Record an MP4 of the trained policy (needs ffmpeg or imageio[ffmpeg])
python scripts/record_video.py --config <cfg> --checkpoint <ckpt>

# Run tests
pytest tests/ -v
```

`WANDB_MODE=disabled` short-circuits W&B logging without editing configs.

---

## Design principles

**One interface for all agents.** `BaseAgent` enforces `select_action / update / save / load`. Adding a new algorithm = subclass + YAML config.

**Configs over magic numbers.** Every hyperparameter lives in `configs/`. Experiments are fully reproducible by sharing the YAML file and seed.

**Separate concerns.** The `Trainer` owns the loop; the agent owns the math. Swapping algorithms never touches the training code.

**Scientific evaluation.** `Evaluator` runs N greedy episodes, reports mean ± std, and exports CSV. Results can also be logged to Weights & Biases.

**Tested core.** `tests/test_core.py` covers buffers, schedulers, and agent save/load so refactors don't silently break things.