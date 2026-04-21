from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sauerbraten_env import SauerbratenEnv
import json

env = SauerbratenEnv(max_steps=50000)

checkpoint = CheckpointCallback(
    save_freq=500000,
    save_path="./checkpoints_cont3/",
    name_prefix="sauerbraten_ppo_cont3"
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs_cont3/",
    n_steps=50000,
    batch_size=500,
    n_epochs=4,
    learning_rate=0.0001,
    ent_coef=0.01,         
    clip_range=0.2,
    )


#model = PPO.load(
#    "./checkpoints_cont/sauerbraten_ppo_cont_9500000_steps",
#    env=env,
#    tensorboard_log="./tb_logs_cont/",
#)

model.learn(total_timesteps=10_000_000, callback=checkpoint)

#model.learn(
#    total_timesteps=10_000_000,  
#    callback=checkpoint,
#    reset_num_timesteps=False    
#)

model.save("sauerbraten_ppo_cont3_final")

# save episode stats
stats = env.get_episode_stats()
with open("episode_stats_cont3.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved {len(stats)} episode stats to episode_stats_cont3.json")