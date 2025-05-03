# -*- coding: utf-8 -*-
"""visualize_policy_all_goals.py

Visualizes and tests the actor's policy across all goal positions in HelicopterControlEnv.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from customised_env import HelicopterControlEnv

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def visualize_policy(
    actor: torch.nn.Module,
    env,
    goal: list,
    max_steps: int = 40,
    use_matplotlib: bool = False,
    log_file=None,
):
    """
    Runs a single episode with the actor for a specific goal, returning performance metrics.
    Optionally uses Matplotlib for visualization.
    Logs details to a file if provided.

    Args:
        actor: Trained actor network.
        env: HelicopterControlEnv instance.
        goal: List [x, y] specifying the goal position.
        max_steps: Maximum steps per episode.
        use_matplotlib: Whether to render with Matplotlib.
        log_file: File handle for logging details.

    Returns:
        dict: Metrics including total_reward, success, steps, and trajectory.
    """
    obs, _ = env.reset(goal=goal)
    total_reward = 0
    trajectory = [obs[:2].copy()]  # Store agent positions
    success = False
    steps = 0

    if use_matplotlib:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(-1, env.grid_size)
        ax.set_ylim(-1, env.grid_size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.plot(env.goal_pos[0], env.goal_pos[1], "r*", label="Goal", markersize=15)
        (line,) = ax.plot([], [], "bo-", label="Helicopter", markersize=10)
        ax.legend()
        ax.set_title(f"Goal: {goal}, Step: 0, Reward: 0.0")

    for step in range(max_steps):
        obs_tensor = (
            torch.from_numpy(obs).to(dtype=torch.float32, device=DEVICE).view(1, -1)
        )
        with torch.no_grad():
            logits = actor(obs_tensor)
            act = logits.argmax(-1).item()

        obs, reward, done, truncated, _ = env.step(act)
        total_reward += reward
        trajectory.append(obs[:2].copy())
        steps += 1

        env.render(mode="human")
        if use_matplotlib:
            trajectory_array = np.array(trajectory)
            line.set_data(trajectory_array[:, 0], trajectory_array[:, 1])
            ax.set_title(f"Goal: {goal}, Step: {step + 1}, Reward: {total_reward:.2f}")
            plt.pause(0.1)

        if log_file:
            log_file.write(
                f"Goal: {goal}, Step: {step + 1}, Action: {act}, "
                f"Reward: {reward:.2f}, Position: {obs[:2]}\n"
            )

        if done:
            success = True
            break
        if truncated:
            break

    if use_matplotlib:
        plt.ioff()
        plt.close()

    if log_file:
        log_file.write(
            f"Goal: {goal}, Episode ended. Total Reward: {total_reward:.2f}, "
            f"Success: {success}, Steps: {steps}\n\n"
        )

    return {
        "goal": goal,
        "total_reward": total_reward,
        "success": success,
        "steps": steps,
        "trajectory": trajectory,
    }


def test_all_goals(
    actor: torch.nn.Module, env, max_steps: int = 40, use_matplotlib: bool = False
):
    """
    Tests the actor's policy across all goal positions in the grid.
    Saves results to a log file and prints a summary.

    Args:
        actor: Trained actor network.
        env: HelicopterControlEnv instance.
        max_steps: Maximum steps per episode.
        use_matplotlib: Whether to render with Matplotlib (slows down testing).
    """
    grid_size = 10
    goal_list = [[x, y] for y in range(grid_size) for x in range(grid_size)]
    results = []

    # Open log file
    with open("policy_test_log.txt", "w") as log_file:
        log_file.write("Policy Test Log\n\n")

        # Test each goal
        for goal in goal_list:
            result = visualize_policy(
                actor, env, goal, max_steps, use_matplotlib, log_file
            )
            results.append(result)

        # Summarize results
        successes = [r["success"] for r in results]
        total_rewards = [r["total_reward"] for r in results]
        steps = [r["steps"] for r in results]
        success_rate = np.mean(successes)
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(steps)
        failed_goals = [r["goal"] for r in results if not r["success"]]

        summary = (
            f"\nSummary:\n"
            f"Total Goals Tested: {len(goal_list)}\n"
            f"Success Rate: {success_rate:.2f}\n"
            f"Average Reward: {avg_reward:.2f}\n"
            f"Average Steps: {avg_steps:.2f}\n"
            f"Failed Goals: {failed_goals}\n"
        )
        log_file.write(summary)
        print(summary)


if __name__ == "__main__":
    from solve import ActorNetwork

    env = HelicopterControlEnv()
    actor = ActorNetwork(
        obs_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
    ).to(DEVICE)

    # Load trained actor weights
    actor.load_state_dict(torch.load("customised_env_actor.pt", map_location=DEVICE))
    actor.eval()

    # Test all goals
    test_all_goals(actor, env, max_steps=40, use_matplotlib=False)
