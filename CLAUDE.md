# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Install (editable; needed so `agents.*`, `common.*`, etc. import from anywhere)
pip install -r requirements.txt
pip install -e .

# Train (config filename's prefix selects the algorithm — see "Agent routing" below)
python scripts/train.py --config configs/reinforce_cartpole.yaml
python scripts/train.py --config configs/dqn_cartpole.yaml --device cuda --seed 123

# Evaluate a checkpoint
python scripts/evaluate.py \
    --config configs/reinforce_cartpole.yaml \
    --checkpoint results/checkpoints/reinforce_cartpole/ReinforceAgent_ep200.pt \
    --n_episodes 20

# Record an MP4 of the trained policy (needs ffmpeg or imageio[ffmpeg])
python scripts/record_video.py --config <cfg> --checkpoint <ckpt>

# Tests
pytest tests/ -v
pytest tests/test_core.py::TestReplayBuffer -v        # single class
pytest tests/test_core.py::TestReplayBuffer::test_push_and_len -v   # single test
```

W&B logging is enabled when a config sets `logging.use_wandb: true`. Set `WANDB_MODE=disabled` to short-circuit it without editing configs.

## Architecture

The repo is organised around a single contract — `BaseAgent` in [agents/base_agent.py](agents/base_agent.py) — and a `Trainer` ([training/trainer.py](training/trainer.py)) that owns the env-interaction loop. Adding a new algorithm is "subclass `BaseAgent` + add a YAML config + register in `scripts/train.py`."

### The agent ↔ trainer split

- `BaseAgent` requires `select_action / update / save / load`, plus optional hooks `on_step_end` (epsilon decay etc.) and `on_episode_end` (per-episode schedules).
- `Trainer` exposes **four distinct training loops**, one per algorithm family. They are NOT interchangeable:
  - `train_on_policy()` — REINFORCE-style: collect a full episode into a `RolloutBuffer`, then one update.
  - `train_actor_critic()` — same shape as on-policy but the agent's `update` consumes Monte-Carlo returns plus a learned baseline.
  - `train_ppo()` — fixed-T rollout that may span episodes, then K-epoch mini-batch update; calls extra agent methods `collect_step`, `finish_rollout`, and a no-arg `update()`. PPO owns its own buffer (`agents/ppo/buffer.py`), not `RolloutBuffer`.
  - `train_off_policy(replay_buffer)` — step → push → sample → update; handles `warmup_steps` of random exploration. Discrete agents return 0-d numpy scalars; the loop unwraps these to `int` while leaving continuous (1-D) action arrays alone — preserve this branch if you touch it ([training/trainer.py:288](training/trainer.py#L288)).

### Agent routing (scripts/train.py)

`scripts/train.py` is the single CLI entry point. It infers the algorithm by splitting the config filename on `_` and taking the first token (`dqn_cartpole.yaml` → `dqn`). Two things must stay in sync when adding an agent:

1. The `build_agent()` registry block — algo name → constructor + which config keys it reads.
2. The dispatch at the bottom of `main()` — algo name → which `Trainer.train_*` loop to call (`reinforce`/`actor` → `train_on_policy`, `ppo` → `train_ppo`, `sarsa` → `train_sarsa`, everything else → `train_off_policy`).

`scripts/evaluate.py` has its own `build_agent` that currently only knows REINFORCE — extend it when evaluating other algorithms.

### Configs are the source of truth

Every hyperparameter lives in `configs/*.yaml`. The three top-level sections — `agent`, `training`, `logging` — map to constructor kwargs, `Trainer` config, and `Logger` kwargs respectively. The `logging.wandb_config` sub-dict is forwarded to `wandb.init`; if `name` is missing, `train.py` auto-generates `<agent>/<env_id>/seed_<seed>/<timestamp>`.

### Shared utilities worth knowing about

- [common/replay_buffer.py](common/replay_buffer.py) — `ReplayBuffer` (off-policy) and `RolloutBuffer` (on-policy, computes discounted MC returns in `.get(gamma=...)`).
- [common/utils.py](common/utils.py) — `build_mlp`, `set_seed`, `load_config`, `soft_update` (Polyak), `hard_update`, `explained_variance`.
- [common/schedulers.py](common/schedulers.py) — `LinearSchedule`, `ExponentialSchedule` (used by DQN/SARSA for ε decay).
- [envs/wrappers.py](envs/wrappers.py) — `make_env(env_id, seed, normalize_obs=..., clip_reward=..., scale_reward=..., record_stats=True)` is the standard factory; `RecordEpisodeStats` is on by default and populates `info["episode"]` at episode end.

### Results / artifacts layout

`results/` is git-ignored. Defaults from configs put checkpoints under `results/checkpoints/<run>/` (filename pattern `<AgentClass>_ep<N>.pt`) and W&B/CSV logs under `results/runs/<run>/`. Eval CSVs end up next to the checkpoint as `eval_results.csv`.
