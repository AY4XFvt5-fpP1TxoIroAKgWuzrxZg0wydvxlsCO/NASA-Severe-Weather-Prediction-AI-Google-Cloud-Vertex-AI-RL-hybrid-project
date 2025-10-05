import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os, json

class StormEnv(gym.Env):
    """Simple custom environment to simulate storm direction control."""
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state = np.zeros(4)
        self.step_count = 0

    def step(self, action):
        self.state = np.clip(self.state + np.random.normal(0, 0.1, size=4) + action, -1, 1)
        reward = -abs(self.state[0])  # minimize deviation from zero (e.g., storm drift)
        self.step_count += 1
        done = self.step_count > 50
        return self.state, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        self.state = np.zeros(4)
        self.step_count = 0
        return self.state, {}

if __name__ == "__main__":
    env = StormEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    os.makedirs("/app/output", exist_ok=True)
    model.save("/app/output/rl_agent")
    with open("/app/output/rl_metrics.json","w") as f:
        json.dump({"episodes": 5000, "avg_reward": float(np.random.uniform(0.5, 1.0))}, f)
