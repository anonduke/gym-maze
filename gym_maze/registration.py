# gym_maze/registration.py
from gymnasium.envs.registration import register

def register_envs():
    register(
        id='maze-sample-5x5-v0',  # Environment ID
        entry_point='gym_maze.envs.maze_env_sample:MazeEnvSample5x5',  # Path to your custom environment class
        max_episode_steps=2000,  # Maximum steps per episode
    )
    # Add other environments here if needed
