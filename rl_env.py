import gym
from gym import spaces
import numpy as np

class WeatherShiftEnv(gym.Env):
    """Simulates storm latitude response to SST/wind perturbations."""
    def __init__(self, target_latitude=25.0):
        super().__init__()
        self.target_lat = target_latitude
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(3, dtype=np.float32)
        return self.state

    def step(self, action):
        sst, su, sv = action
        simulated_lat = 20 + 1.2*sst + 0.08*(su + sv) + np.random.normal(0, 0.2)
        reward = -abs(simulated_lat - self.target_lat)
        return self.state, reward, True, {"simulated_lat": simulated_lat}
