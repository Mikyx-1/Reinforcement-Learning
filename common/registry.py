"""
Single source of truth for turning a config dict into an agent instance.

Shared by scripts/train.py, scripts/evaluate.py, and scripts/record_video.py so
they never drift out of sync on which algorithms are supported.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym

from agents.base_agent import BaseAgent

# Config filename stem (before the first "_") -> agent name understood here.
AGENT_NAMES = ("reinforce", "actor", "ppo", "gnnppo", "dqn", "sarsa", "ddpg", "td3", "sac")


def infer_agent_name(config_path: str | Path) -> str:
    """Guess agent name from config filename, e.g. 'reinforce_cartpole.yaml' -> 'reinforce'."""
    return Path(config_path).stem.split("_")[0].lower()


def build_agent(name: str, env: gym.Env, cfg: dict[str, Any], device: str = "cpu") -> BaseAgent:
    """Instantiate the agent named `name` (see AGENT_NAMES) from a loaded YAML config."""
    agent_cfg = cfg["agent"]

    if name == "reinforce":
        from agents.reinforce.agent import ReinforceAgent

        return ReinforceAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            lr=agent_cfg["lr"],
            gamma=agent_cfg["gamma"],
            entropy_coef=agent_cfg.get("entropy_coef", 0.0),
            device=device,
        )

    if name == "actor":
        from agents.actor_critic.agent import ActorCriticAgent

        return ActorCriticAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            lr=agent_cfg["lr"],
            gamma=agent_cfg["gamma"],
            ent_coef=agent_cfg.get("entropy_coef", 0.0),
            device=device,
        )

    if name == "ppo":
        from agents.ppo.agent import PPOAgent

        return PPOAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            lr=agent_cfg["lr"],
            gamma=agent_cfg["gamma"],
            lam=agent_cfg.get("lam", 0.95),
            clip_eps=agent_cfg.get("clip_eps", 0.2),
            n_epochs=agent_cfg.get("n_epochs", 10),
            batch_size=agent_cfg.get("batch_size", 64),
            rollout_steps=agent_cfg.get("rollout_steps", 2048),
            vf_coef=agent_cfg.get("vf_coef", 0.5),
            ent_coef=agent_cfg.get("ent_coef", 0.01),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            clip_value_loss=agent_cfg.get("clip_value_loss", True),
            target_kl=agent_cfg.get("target_kl", 0.015),
            device=device,
        )

    if name == "gnnppo":
        from agents.gnn_ppo.agent import GNNPPOAgent

        return GNNPPOAgent(
            env=env,
            hidden_dim=agent_cfg.get("hidden_dim", 64),
            n_layers=agent_cfg.get("n_layers", 2),
            lr=agent_cfg["lr"],
            gamma=agent_cfg["gamma"],
            lam=agent_cfg.get("lam", 0.95),
            clip_eps=agent_cfg.get("clip_eps", 0.2),
            n_epochs=agent_cfg.get("n_epochs", 10),
            batch_size=agent_cfg.get("batch_size", 64),
            rollout_steps=agent_cfg.get("rollout_steps", 2048),
            vf_coef=agent_cfg.get("vf_coef", 0.5),
            ent_coef=agent_cfg.get("ent_coef", 0.01),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            clip_value_loss=agent_cfg.get("clip_value_loss", True),
            target_kl=agent_cfg.get("target_kl", 0.015),
            device=device,
        )

    if name == "dqn":
        from agents.dqn.agent import DQNAgent

        return DQNAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            lr=agent_cfg["lr"],
            gamma=agent_cfg["gamma"],
            eps_start=agent_cfg.get("eps_start", 1.0),
            eps_end=agent_cfg.get("eps_end", 0.01),
            eps_decay_steps=agent_cfg.get("eps_decay_steps", 500),
            target_update_freq=agent_cfg.get("target_update_freq", 1),
            use_double=agent_cfg.get("use_double", True),
            use_dueling=agent_cfg.get("use_dueling", False),
            device=device,
        )

    if name == "sarsa":
        from agents.sarsa.agent import SarsaAgent

        return SarsaAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            lr=agent_cfg["lr"],
            gamma=agent_cfg["gamma"],
            eps_start=agent_cfg.get("eps_start", 1.0),
            eps_end=agent_cfg.get("eps_end", 0.01),
            eps_decay_steps=agent_cfg.get("eps_decay_steps", 500),
            tau=agent_cfg.get("tau", 0.005),
            device=device,
        )

    if name == "ddpg":
        from agents.ddpg.agent import DDPGAgent

        return DDPGAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            actor_lr=agent_cfg["actor_lr"],
            critic_lr=agent_cfg["critic_lr"],
            gamma=agent_cfg["gamma"],
            tau=agent_cfg["tau"],
            noise_type=agent_cfg["noise_type"],
            noise_sigma=agent_cfg["noise_sigma"],
            noise_sigma_min=agent_cfg.get("noise_sigma_min", 0.02),
            noise_sigma_decay=agent_cfg.get("noise_sigma_decay", 0.999),
            device=device,
        )

    if name == "td3":
        from agents.td3.agent import TD3Agent

        return TD3Agent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            actor_lr=agent_cfg["actor_lr"],
            critic_lr=agent_cfg["critic_lr"],
            gamma=agent_cfg["gamma"],
            tau=agent_cfg["tau"],
            policy_delay=agent_cfg.get("policy_delay", 2),
            target_noise_sigma=agent_cfg.get("target_noise_sigma", 0.2),
            target_noise_clip=agent_cfg.get("target_noise_clip", 0.5),
            noise_type=agent_cfg.get("noise_type", "gaussian"),
            noise_sigma=agent_cfg.get("noise_sigma", 0.1),
            noise_sigma_min=agent_cfg.get("noise_sigma_min", 0.01),
            noise_sigma_decay=agent_cfg.get("noise_sigma_decay", 1.0),
            device=device,
        )

    if name == "sac":
        from agents.sac.agent import SACAgent

        return SACAgent(
            env=env,
            hidden_dims=agent_cfg["hidden_dims"],
            actor_lr=agent_cfg["actor_lr"],
            critic_lr=agent_cfg["critic_lr"],
            alpha_lr=agent_cfg.get("alpha_lr", 3e-4),
            gamma=agent_cfg["gamma"],
            tau=agent_cfg["tau"],
            init_alpha=agent_cfg.get("init_alpha", 0.2),
            autotune_alpha=agent_cfg.get("autotune_alpha", True),
            target_entropy=agent_cfg.get("target_entropy", None),
            device=device,
        )

    raise ValueError(f"Unknown agent '{name}'. Register it in common/registry.py.")
