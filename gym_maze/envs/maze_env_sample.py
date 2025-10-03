# gym_maze/envs/maze_env_sample.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnvSample5x5(gym.Env):
    def __init__(self):
        super(MazeEnvSample5x5, self).__init__()
        
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int32)

        self.state = np.zeros((5, 5))  # 5x5 maze initialized to zeros
        self.agent_position = (0, 0)  # Starting position at (0, 0)
        self.state[self.agent_position] = 1  # Mark the agent's position

    def reset(self, seed=None, options=None):
        # Handle the seed and options arguments
        if seed is not None:
            np.random.seed(seed)
        
        self.state = np.zeros((5, 5))  # Reset the maze to all zeros
        self.agent_position = (0, 0)   # Reset the agent's position
        self.state[self.agent_position] = 1  # Set the agent's position in the maze

        return self.state, {}

    def step(self, action):
        # Define movements: 0=Up, 1=Down, 2=Left, 3=Right
        x, y = self.agent_position

        if action == 0:  # Up
            if x > 0:
                x -= 1
        elif action == 1:  # Down
            if x < 4:
                x += 1
        elif action == 2:  # Left
            if y > 0:
                y -= 1
        elif action == 3:  # Right
            if y < 4:
                y += 1

        self.agent_position = (x, y)
        self.state = np.zeros((5, 5))
        self.state[self.agent_position] = 1  # Update agent's position in the maze
        
        # Check if agent reached the goal (bottom-right corner)
        done = (self.agent_position == (4, 4))
        reward = 1 if done else -0.1  # Reward: 1 for reaching the goal, -0.1 for each step
        
        truncated = False  # In this simple case, we don't use truncation, so it's False
        info = {}  # You can add additional information here if needed
        
        # Return all 5 values (observation, reward, done, truncated, info)
        return self.state, reward, done, truncated, info

    def render(self):
        print(self.state)
