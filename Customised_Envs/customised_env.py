import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


class HelicopterControlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 50
        self.max_step_size = 150
        self.action_space = spaces.Discrete(5)  # up, left, right, up-left, up-right
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32
        )
        self.gravity = 3
        self.step_size = 5

        self.agent_pos = np.array([0, 0])  # (x, y)
        self.goal_pos = np.array([0, 0])
        self.step_count = 0

    def reset(self, goal=None, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])

        self.goal_pos = np.array(
            [
                np.random.randint(0, self.grid_size - 1),
                np.random.randint(0, self.grid_size - 1),
            ]
        )
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)

    def step(self, action):
        self.step_count += 1
        # old_agent_pos = self.agent_pos.copy()  # Store old position

        if action == 0:  # thrust up
            self.agent_pos[1] += self.step_size
        elif action == 1:  # thrust left
            self.agent_pos[0] -= self.step_size
        elif action == 2:  # thrust right
            self.agent_pos[0] += self.step_size
        elif action == 3:  # thrust up-left
            self.agent_pos += [-self.step_size, self.step_size]
        elif action == 4:  # thrust up-right
            self.agent_pos += [self.step_size, self.step_size]

        self.agent_pos[1] -= self.gravity
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

        done = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.step_count >= self.max_step_size

        if done:
            reward = 1000  # Reduced sparse reward
            print("The agent has hit the goal!")
        else:
            distance = np.sum(np.abs(self.agent_pos - self.goal_pos))
            reward = -1 - 0.5 * distance  # Increased dense penalty

        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode="human"):
        if mode == "human":
            grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
            x, y = self.agent_pos
            gx, gy = self.goal_pos
            grid[int(y), int(x)] = "H"
            grid[int(gy), int(gx)] = "G"
            print("\n".join("".join(row) for row in grid[::-1]))
            print()
        elif mode == "rgb_array":
            # For potential future use with other rendering frameworks
            pass


# Visualization function using Matplotlib
def plot_environment(env, step, total_reward, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()
    ax.set_xlim(-1, env.grid_size)
    ax.set_ylim(-1, env.grid_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Plot agent and goal
    ax.plot(env.agent_pos[0], env.agent_pos[1], "bo", label="Helicopter", markersize=10)
    ax.plot(env.goal_pos[0], env.goal_pos[1], "r*", label="Goal", markersize=15)
    ax.legend()
    ax.set_title(f"Step: {step}, Total Reward: {total_reward:.2f}")
    plt.pause(0.1)  # Brief pause to update the plot


# Main visualization loop
def visualize_environment(use_matplotlib=True):
    env = HelicopterControlEnv()
    observation, info = env.reset(seed=42)

    # Set up Matplotlib if used
    if use_matplotlib:
        plt.ion()  # Interactive mode for dynamic updates
        fig, ax = plt.subplots()

    total_reward = 0
    for step in range(1000):
        action = env.action_space.sample()  # Random policy
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Render environment
        if use_matplotlib:
            plot_environment(env, step, total_reward, ax)
        else:
            env.render()  # Text-based rendering
            time.sleep(0.1)  # Slow down for readability

        # Reset if episode ends
        if terminated or truncated:
            print(
                f"Episode ended. Terminated: {terminated}, Truncated: {truncated}, Total Reward: {total_reward}"
            )
            # observation, info = env.reset()
            # if use_matplotlib:
            #     plot_environment(env, step + 1, 0, ax)  # Show reset state
            break

        if use_matplotlib:
            plt.draw()

    env.close()
    if use_matplotlib:
        plt.ioff()
        plt.show()


# Run the visualization
if __name__ == "__main__":
    visualize_environment(use_matplotlib=True)  # Set to False for text-based rendering
