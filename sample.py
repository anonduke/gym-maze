import os
import gym
from gym.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# --- Register the Maze environment (old Gym API) ---
register(
    id="CustomMaze-v0",
    entry_point="gym_maze.envs.maze_env:MazeEnv",
)

# --- Absolute path to a maze file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
maze_path = os.path.join(BASE_DIR, "gym_maze", "envs", "maze_samples", "maze2d_10x10.npy")

# --- Create environment ---
env = gym.make("CustomMaze-v0", maze_file=maze_path)
n_actions = env.action_space.n  # expected 4

# ===== Robust state extraction =====
def get_agent_pos_from_obs(obs: np.ndarray):
    """
    Try common encodings first: agent labeled as 2 (or 3/4/1) in a 2D grid.
    Return (row, col) if found, else None.
    """
    if isinstance(obs, np.ndarray) and obs.ndim == 2:
        for agent_val in (2, 3, 4, 1):
            locs = np.argwhere(obs == agent_val)
            if locs.size > 0:
                r, c = locs[0]
                return (int(r), int(c))
    return None

def get_agent_pos_from_env(env):
    """
    Try common internal attributes on unwrapped env.
    Return (row, col) if found, else None.
    """
    un = env.unwrapped
    candidates = ("agent_pos", "_agent_pos", "player_pos", "player")
    for name in candidates:
        if hasattr(un, name):
            v = getattr(un, name)
            if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2:
                return (int(v[0]), int(v[1]))
    return None

def state_from_obs_or_env(env, obs):
    """
    Preferred: (row, col). Fallback: hash of observation bytes (works even if agent not in obs).
    Returns (state, is_coord_state: bool)
    """
    pos = get_agent_pos_from_obs(obs)
    if pos is not None:
        return pos, True

    pos = get_agent_pos_from_env(env)
    if pos is not None:
        return pos, True

    # Fallback: use hashed observation as state (string key).
    # Works for learning even if agent isn’t encoded in obs.
    if isinstance(obs, np.ndarray):
        return ("OBS_HASH", hash(obs.tobytes())), False
    # final fallback
    return ("OBS_REPR", hash(str(obs))), False

# ===== Policy / Q-learning =====
Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

def epsilon_greedy_action(state, epsilon: float):
    if random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))

episodes = 400
gamma = 0.99
alpha = 0.5
epsilon_start, epsilon_end = 0.9, 0.05
epsilon_decay_episodes = int(0.7 * episodes)

def epsilon_schedule(ep):
    if ep >= epsilon_decay_episodes:
        return epsilon_end
    return epsilon_start - (ep / epsilon_decay_episodes) * (epsilon_start - epsilon_end)

returns = []
coord_state_seen = False

for ep in range(episodes):
    obs = env.reset()  # old Gym: obs only
    state, is_coord = state_from_obs_or_env(env, obs)
    coord_state_seen = coord_state_seen or is_coord

    done = False
    ep_return = 0.0
    eps = epsilon_schedule(ep)
    steps, max_steps = 0, (2000 if (isinstance(obs, np.ndarray) and obs.shape[0] > 20) else 600)

    while not done and steps < max_steps:
        a = epsilon_greedy_action(state, eps)
        next_obs, r, done, info = env.step(a)
        next_state, _ = state_from_obs_or_env(env, next_obs)

        best_next = np.max(Q[next_state])
        td_target = r + (0.0 if done else gamma * best_next)
        Q[state][a] += alpha * (td_target - Q[state][a])

        state = next_state
        obs = next_obs
        ep_return += r
        steps += 1

    returns.append(ep_return)
    if (ep + 1) % max(1, episodes // 10) == 0:
        print(f"Episode {ep+1}/{episodes} | ε={eps:.3f} | return={ep_return:.2f} | coord_state={coord_state_seen}")

# ===== Greedy rollout (visual) =====
plt.ion()
fig, ax = plt.subplots()
obs = env.reset()
state, _ = state_from_obs_or_env(env, obs)
done = False
frames = 0
max_frames = 2000 if (isinstance(obs, np.ndarray) and obs.shape[0] > 20) else 800

while not done and frames < max_frames:
    a = int(np.argmax(Q[state]))
    obs, r, done, info = env.step(a)
    frame = env.render(mode="ansi")  # image array
    ax.clear()
    ax.imshow(frame)
    ax.set_title(f"Greedy rollout | step {frames} | reward {r:.2f}")
    plt.pause(0.01)
    state, _ = state_from_obs_or_env(env, obs)
    frames += 1

env.close()
plt.ioff()
plt.show()

# Optional: training curve
if len(returns) > 10:
    def moving_avg(x, k=20):
        out, s = [], 0.0
        from collections import deque
        q = deque(maxlen=k)
        for v in x:
            q.append(v); s = sum(q); out.append(s/len(q))
        return out
    plt.figure()
    plt.plot(returns, alpha=0.4, label="Return")
    plt.plot(moving_avg(returns, 20), label="Return (MA20)")
    plt.legend(); plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("Training progress")
    plt.show()
