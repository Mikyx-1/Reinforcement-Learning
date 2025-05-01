import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HelicopterControlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 50  # Reduced grid size
        self.max_step_size = 100  # Reduced max steps
        self.action_space = spaces.Discrete(5)  # up, left, right, up-left, up-right
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32
        )
        self.gravity = 0.5  # Reduced gravity
        self.step_size = 2  # Increased movement step size

        self.agent_pos = np.array([0, 0])  # (x, y)
        self.goal_pos = np.array([0, 0])
        self.step_count = 0  # Initialize step counter

    def reset(self, goal=None, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding
        # Initialize agent and goal in a smaller region (e.g., within 30x30)
        self.agent_pos = np.array([np.random.randint(0, self.grid_size-1), np.random.randint(0, self.grid_size-1)])
        if goal is None:
            self.goal_pos = np.array(
                [np.random.randint(0, self.grid_size-1), np.random.randint(0, self.grid_size-1)]
            )
        else:
            self.goal_pos = np.array(goal)
        self.step_count = 0  # Reset step counter
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)

    def step(self, action):
        self.step_count += 1  # Increment step counter

        # Apply action with increased step size
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

        # Apply gravity (pull down)
        self.agent_pos[1] -= self.gravity

        # Clip to grid bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

        # Compute reward and termination
        done = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.step_count >= self.max_step_size

        # Reward shaping: sparse + dense (negative Manhattan distance)
        if done:
            reward = 100
        else:
            distance = np.sum(np.abs(self.agent_pos - self.goal_pos))
            reward = -1 - 0.1 * distance  # Dense penalty for distance

        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        grid[int(y), int(x)] = "H"  # Cast to int for indexing
        grid[int(gy), int(gx)] = "G"
        print("\n".join("".join(row) for row in grid[::-1]))
        print()


env = HelicopterControlEnv()
obs, _ = env.reset()
print(f"obs: {obs}")
