import numpy as np
import random

# Simple Custom Environment (S=15 states, A=11 actions)
class SimpleEnv:
    def __init__(self):
        self.n_states = 15  # S > 10
        self.n_actions = 11  # A > 10
        self.terminal_state = 14
        self.reset()

    def reset(self):
        self.state = 0  # Start at state 0
        return self.state

    def step(self, action):
        # Random transition: Move +1 to +5 states based on action (0-10), but cap at terminal
        next_state = min(self.state + (action % 5 + 1), self.terminal_state)
        reward = random.uniform(-1, 1)  # Random reward -1 to +1 per step
        if next_state == self.terminal_state:
            reward = 10  # Big reward at end
            done = True
        else:
            done = False
        self.state = next_state
        return next_state, reward, done

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))  # Q-table initialized to 0

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore: random action
        else:
            return np.argmax(self.Q[state])  # Exploit: best action

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward  # No future if terminal
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

# Run the code
env = SimpleEnv()
agent = QLearningAgent(env.n_states, env.n_actions)

num_episodes = 100  # Run 100 episodes
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        state = next_state
    print(f"Episode {episode + 1} finished.")

print("Trained Q-table:\n", agent.Q)  # Show the learned table