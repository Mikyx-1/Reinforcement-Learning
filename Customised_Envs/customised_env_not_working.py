import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HelicopterControlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 100
        self.max_step_size = 250
        self.action_space = spaces.Discrete(5)  # up, left, right, up-left, up-right
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32
        )
        self.gravity = 1  # cells pulled down per step

        self.agent_pos = np.array([0, 0])  # (x, y)
        self.goal_pos = np.array([0, 0])
        self.step_count = 0  # Initialize step counter

    def reset(self, goal=None, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding for reproducibility
        self.agent_pos = np.array(
            [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
        )
        if goal is None:
            self.goal_pos = np.array(
                [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            )
        else:
            self.goal_pos = np.array(goal)
        self.step_count = 0  # Reset step counter
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)

    def step(self, action):
        self.step_count += 1  # Increment step counter

        # Apply action
        if action == 0:  # thrust up
            self.agent_pos[1] += 1
        elif action == 1:  # thrust left
            self.agent_pos[0] -= 1
        elif action == 2:  # thrust right
            self.agent_pos[0] += 1
        elif action == 3:  # thrust up-left
            self.agent_pos += [-1, 1]
        elif action == 4:  # thrust up-right
            self.agent_pos += [1, 1]

        # Apply gravity (pull down)
        self.agent_pos[1] -= self.gravity

        # Clip to grid bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

        # Compute reward and termination
        done = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.step_count >= self.max_step_size
        reward = 100 if done else -1  # sparse reward

        # Return observation, reward, terminated, truncated, info
        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        grid[y, x] = "H"
        grid[gy, gx] = "G"
        print("\n".join("".join(row) for row in grid[::-1]))
        print()


env = HelicopterControlEnv()
obs = env.reset()
print(f"obs: {obs}")
