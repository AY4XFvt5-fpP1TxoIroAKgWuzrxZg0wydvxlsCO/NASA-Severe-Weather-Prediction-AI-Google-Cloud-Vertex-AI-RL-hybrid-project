from stable_baselines3 import PPO
from rl.rl_env import WeatherShiftEnv

def train_rl(episodes=2000):
    env = WeatherShiftEnv(target_latitude=26.0)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=episodes)
    model.save("rl_weather_agent")
    print("RL agent saved as rl_weather_agent.zip")

if __name__ == "__main__":
    train_rl()
