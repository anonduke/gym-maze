import numpy as np
import matplotlib.pyplot as plt
import random

# Simple 5-state environment
class RandomWalkEnv:
    def __init__(self):
        self.states = [0,1,2,3,4]  # 0=A (terminal -1), 4=E (terminal +1), start at 2=C
        self.reset()

    def reset(self):
        self.state = 2
        return self.state

    def step(self, action):  # Action 0=left, 1=right
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        if self.state == 0:
            return 0, -1, True
        if self.state == 4:
            return 4, 1, True
        return self.state, 0, False

# MC Method
def mc_value_estimates(num_runs=1000, alpha=0.1, gamma=0.9):
    values = np.zeros(5)  # Value for each state
    for _ in range(num_runs):
        env = RandomWalkEnv()
        state = env.reset()
        episode = []  # Store states, rewards
        done = False
        while not done:
            action = random.choice([0,1])  # Random policy
            next_state, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_state
        # Update at end: Compute return G
        G = 0
        for s, r in reversed(episode):
            G = r + gamma * G
            values[s] += alpha * (G - values[s])
    return values

# Q-Learning (TD) - Simplified for value estimation (similar to TD(0))
def td_value_estimates(num_runs=1000, alpha=0.1, gamma=0.9):
    values = np.zeros(5)
    for _ in range(num_runs):
        env = RandomWalkEnv()
        state = env.reset()
        done = False
        while not done:
            action = random.choice([0,1])
            next_state, reward, done = env.step(action)
            if done:
                target = reward
            else:
                target = reward + gamma * values[next_state]
            values[state] += alpha * (target - values[state])
            state = next_state
    return values

# Run and plot variance (run multiple times, compute std dev)
num_experiments = 50
mc_results = [mc_value_estimates() for _ in range(num_experiments)]
td_results = [td_value_estimates() for _ in range(num_experiments)]

mc_var = np.std(mc_results, axis=0)
td_var = np.std(td_results, axis=0)

plt.bar(range(5), mc_var, alpha=0.5, label='MC Variance')
plt.bar(range(5), td_var, alpha=0.5, label='TD (Q-Learning style) Variance')
plt.xlabel('States')
plt.ylabel('Variance')
plt.legend()
plt.title('Variance Comparison: MC vs TD')
plt.show()