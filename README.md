# Overview

This project trains a reinforcement learning agent to compete against Sauerbraten's built-in hardcoded bot AI using Proximal Policy Optimization (PPO). The agent controls the game's `player1` character through a TCP socket bridge connecting the C++ game engine to a Python training environment.
---

## Repository Structure

```
rl_sauerbraten/
│
├── fpsgame/                        # Modified Sauerbraten C++ source files
│   ├── fps.cpp                     # Main game loop with RL bridge integration
│   └── rl_bridge.h                 # TCP socket bridge (C++ → Python)
│
├── tb_logs_cont2/                  # TensorBoard logs — Experiment B (skill 100)
├── tb_logs_cont3/                  # TensorBoard logs — Experiment A (skill 50)
│
├── thesis_graphs/                  # Generated graphs used in the thesis
│
├── sauerbraten_env.py              # Gymnasium environment wrapper
├── train.py                        # PPO training script
├── evaluate.py                     # Evaluation script
│
├── episode_stats_cont2.json        # Per-episode stats — Experiment B (skill 100)
├── episode_stats_cont3.json        # Per-episode stats — Experiment A (skill 50)
│
├── sauerbraten_ppo_cont2_final.zip # Trained model — Experiment B (skill 100)
├── sauerbraten_ppo_cont3_final.zip # Trained model — Experiment A (skill 50)
│
└── README.md
```
## Full Game Source

This project requires a modified build of Cube 2: Sauerbraten. The full, unmodified game source can be found at (https://github.com/embeddedc/sauerbraten). To use this RL agent, clone that repository, replace `src/fpsgame/fps.cpp` with the modified version from this repo, and add `fpsgame/rl_bridge.h` from this repo into the same directory, then rebuild the game. All other game files remain unchanged.

**Video of the agent**: https://youtu.be/QuXSx8AFpmo
