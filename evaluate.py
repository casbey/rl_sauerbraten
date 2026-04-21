from stable_baselines3 import PPO
from sauerbraten_env import SauerbratenEnv
import numpy as np

smooth_yaw = 0.0
env = SauerbratenEnv(max_steps=50000)

model = PPO.load(
    "./checkpoints_cont3/sauerbraten_ppo_cont3_7500000_steps",
    env=env
)

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=False)
    smooth_yaw = smooth_yaw * 0.97 + action[2] * 0.03
    action[2] = smooth_yaw
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        smooth_yaw = 0.0  
        obs, _ = env.reset()