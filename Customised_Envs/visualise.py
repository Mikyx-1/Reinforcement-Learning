# -*- coding: utf-8 -*-
"""visualize_ppo_performance.py

Script to load a trained PPO actor model and visualize its performance in HelicopterControlEnv.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from customised_env import HelicopterControlEnv
from torch.distributions.categorical import Categorical

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


####################################
# Positional Embedding Module
####################################
class PositionalEmbedding2D(torch.nn.Module):
    def __init__(self, grid_size: int, embed_dim: int):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim  # Total embedding size per position
        self.half_dim = (
            embed_dim // 4
        )  # Number of frequency terms per coordinate (x or y)

        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D encoding"

        # Precompute sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, self.half_dim, 1) * (-np.log(10000.0) / self.half_dim)
        ).to(DEVICE)
        self.register_buffer("div_term", div_term)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: Tensor of shape (batch_size, 2) with [x, y] coordinates
        Returns: Tensor of shape (batch_size, embed_dim) with positional embeddings
        """
        batch_size = positions.size(0)
        x, y = positions[:, 0], positions[:, 1]  # Shape: (batch_size,)

        # Initialize embedding tensor
        pe = torch.zeros(batch_size, self.embed_dim, device=DEVICE)

        # X-coordinate encodings
        pos_x = x.unsqueeze(-1)  # (batch_size, 1)
        pe[:, 0 : self.half_dim] = torch.sin(pos_x * self.div_term)
        pe[:, self.half_dim : self.half_dim * 2] = torch.cos(pos_x * self.div_term)

        # Y-coordinate encodings
        pos_y = y.unsqueeze(-1)  # (batch_size, 1)
        pe[:, self.half_dim * 2 : self.half_dim * 3] = torch.sin(pos_y * self.div_term)
        pe[:, self.half_dim * 3 :] = torch.cos(pos_y * self.div_term)

        return pe


####################################
# Actor Network with Positional Embeddings
####################################
class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        obs_space_size: int,
        action_space_size: int,
        grid_size: int = 50,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Positional embedding for agent and goal
        self.pos_embed = PositionalEmbedding2D(
            grid_size, embed_dim // 2
        )  # embed_dim // 2 per position

        # Input size: agent embedding + goal embedding + raw obs
        input_dim = (
            embed_dim // 2
        ) * 2 + obs_space_size  # Agent and goal embeddings + raw obs
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, action_space_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: Tensor of shape (batch_size, 4) with [x_agent, y_agent, x_goal, y_goal]
        Returns: Action logits
        """
        # Extract agent and goal positions
        agent_pos = obs[:, :2]  # (batch_size, 2)
        goal_pos = obs[:, 2:]  # (batch_size, 2)

        # Get positional embeddings
        agent_pe = self.pos_embed(agent_pos)  # (batch_size, embed_dim // 2)
        goal_pe = self.pos_embed(goal_pos)  # (batch_size, embed_dim // 2)

        # Concatenate embeddings and raw observations
        input_features = torch.cat(
            [agent_pe, goal_pe, obs], dim=-1
        )  # (batch_size, embed_dim + obs_space_size)

        return self.network(input_features)


####################################
# Visualization Function
####################################
def visualize_policy(actor: torch.nn.Module, env, max_steps=150, use_matplotlib=True):
    """
    Runs a single episode with the actor, rendering the environment.
    Optionally uses Matplotlib for dynamic visualization.
    """
    obs, _ = env.reset()
    total_reward = 0
    trajectory = [obs[:2].copy()]  # Store agent positions for plotting

    if use_matplotlib:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots()
        ax.set_xlim(-1, env.grid_size)
        ax.set_ylim(-1, env.grid_size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.plot(env.goal_pos[0], env.goal_pos[1], "r*", label="Goal", markersize=15)
        (line,) = ax.plot([], [], "bo-", label="Helicopter", markersize=10)
        ax.legend()
        ax.set_title(f"Step: 0, Reward: 0.0")

    for step in range(max_steps):
        # Convert observation to tensor
        obs_tensor = (
            torch.from_numpy(obs).to(dtype=torch.float32, device=DEVICE).view(1, -1)
        )

        # Get action from actor
        with torch.no_grad():
            logits = actor(obs_tensor)
            # dist = Categorical(logits=logits)
            # act = dist.sample().item()
            act = logits.argmax(-1).item()

        # Step in environment
        obs, reward, done, truncated, _ = env.step(act)
        total_reward += reward
        trajectory.append(obs[:2].copy())  # Store new agent position

        # Render environment
        env.render(mode="human")  # Text-based rendering

        # Optional Matplotlib rendering
        if use_matplotlib:
            trajectory_array = np.array(trajectory)
            line.set_data(trajectory_array[:, 0], trajectory_array[:, 1])
            ax.set_title(f"Step: {step + 1}, Reward: {total_reward:.2f}")
            plt.pause(0.1)  # Pause to update plot

        print(
            f"Step {step + 1}, Action: {act}, Reward: {reward:.2f}, Position: {obs[:2]}"
        )

        if done or truncated:
            print(
                f"Episode ended. Total Reward: {total_reward:.2f}, Done: {done}, Truncated: {truncated}"
            )
            break

    if use_matplotlib:
        plt.ioff()
        plt.show()


####################################
# Main Execution
####################################
if __name__ == "__main__":
    # Initialize environment
    env = HelicopterControlEnv()
    obs_space = env.observation_space.shape[0]  # 4
    action_space = env.action_space.n  # 4
    grid_size = env.grid_size  # 50

    # Initialize actor model
    actor = ActorNetwork(
        obs_space, action_space, grid_size=grid_size, embed_dim=128
    ).to(DEVICE)

    # Load saved weights
    try:
        actor.load_state_dict(
            torch.load("customised_env_actor.pt", map_location=DEVICE)
        )
        print("Successfully loaded actor weights from 'customised_env_actor.pt'")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit(1)

    # Set actor to evaluation mode
    actor.eval()

    # Run visualization
    print("Starting visualization...")
    visualize_policy(actor, env, max_steps=150, use_matplotlib=True)
