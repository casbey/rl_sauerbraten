import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SauerbratenEnv(gym.Env):

    def __init__(self, host="127.0.0.1", port=42000, max_steps=50000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.steps = 0
        self.max_steps = max_steps

        # tracking
        self.last_frags = 0
        self.last_deaths = 0
        self.last_dist = 999.0
        self.last_enemy_visible = 0
        self.shots_fired = 0
        self.enemy_visible_steps = 0
        self.current_frags = 0
        self.current_deaths = 0
        self.episode_count = 0
        self.episode_start_frags = 0
        self.episode_start_deaths = 0
        self.episode_stats = []

        # socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print("waiting")

        # 25 observations:
        # 0-2:   pos_x, pos_y, pos_z
        # 3-4:   yaw, pitch
        # 5-6:   health, ammo
        # 7-9:   enemy_visible, enemy_dist, enemy_angle_diff
        # 10:    enemy_health
        # 11-13: vel_x, vel_y, vel_z
        # 14:    onground
        # 15:    num_enemies
        # 16:    blocked
        # 17-24: ray0..ray7 (8 raycasts, normalized 0-1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        # Continuous action space:
        # dim 0: move_forward_back  (-1 = back, +1 = forward)
        # dim 1: strafe             (-1 = left, +1 = right)
        # dim 2: yaw_delta          (-1 to +1, 15 deg)
        # dim 3: shoot              ( >=0.5 = shoot)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _connect(self):
        self.conn, addr = self.sock.accept()
        print("connected")

    def _recv_state(self):
        data = b""
        while not data.endswith(b"\n"):
            chunk = self.conn.recv(4096)
            if not chunk:
                raise ConnectionError("disconnected")
            data += chunk
        return json.loads(data.decode())

    def _send_action(self, action_json: str):
        self.conn.sendall((action_json + "\n").encode())

    def _encode_action(self, action):

        move_fb  = float(action[0])   # forward/back
        strafe   = float(action[1])   # strafe
        yaw_d    = float(action[2])   # yaw delta (-1 to 1)
        shoot    = float(action[3])   # shoot threshold

        return json.dumps({
            "move_fb":  round(move_fb,  3),
            "strafe":   round(strafe,   3),
            "yaw_d":    round(yaw_d,    3),
            "shoot":    round(shoot,    3)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # log previous episode stats
        if self.steps > 0:
            episode_frags  = self.current_frags  - self.episode_start_frags
            episode_deaths = self.current_deaths - self.episode_start_deaths
            kd = episode_frags / max(1, episode_deaths)
            self.episode_stats.append({
                "episode":             self.episode_count,
                "frags":               episode_frags,
                "deaths":              episode_deaths,
                "kd_ratio":            round(kd, 2),
                "survival_steps":      self.steps,
                "shots_fired":         self.shots_fired,
                "enemy_visible_steps": self.enemy_visible_steps,
            })
            print(f"[RL] Episode {self.episode_count} stats — "
                  f"frags: {episode_frags}, "
                  f"deaths: {episode_deaths}, "
                  f"k/d: {kd:.2f}, "
                  f"shots: {self.shots_fired}, "
                  f"enemy_visible: {self.enemy_visible_steps}")

        # reset counters
        self.steps              = 0
        self.last_dist          = 999.0
        self.last_enemy_visible = 0
        self.shots_fired        = 0
        self.enemy_visible_steps = 0
        self.episode_count     += 1

        if self.conn is None:
            print("waiting for game to connect...")
            self._connect()

        # send reset signal
        self._send_action('{"reset":1}')
        state = self._recv_state()

        # sync to current game values
        self.last_frags          = state.get("frags",  0)
        self.last_deaths         = state.get("deaths", 0)
        self.current_frags       = self.last_frags
        self.current_deaths      = self.last_deaths
        self.episode_start_frags  = self.last_frags
        self.episode_start_deaths = self.last_deaths

        return self._parse_obs(state), {}

    def step(self, action):
        action_str = self._encode_action(action)
        self._send_action(action_str)
        state = self._recv_state()

        # track shots
        if float(action[3]) >= 0.5:
            self.shots_fired += 1

        # track visibility
        if state.get("enemy_visible", 0):
            self.enemy_visible_steps += 1

        # update stats
        self.current_frags  = state.get("frags",  0)
        self.current_deaths = state.get("deaths", 0)

        obs     = self._parse_obs(state)
        reward  = self._compute_reward(state, action)

        self.steps += 1
        terminated = False
        truncated  = self.steps >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _parse_obs(self, state):
        rays = [state.get(f"ray{i}", 1.0) for i in range(8)]
        return np.array([
            state.get("pos_x",            0),
            state.get("pos_y",            0),
            state.get("pos_z",            0),
            state.get("yaw",              0),
            state.get("pitch",            0),
            state.get("health",         100),
            state.get("ammo",            10),
            float(state.get("enemy_visible", False)),
            state.get("enemy_dist",      999),
            state.get("enemy_angle_diff",  0),
            state.get("enemy_health",    100),
            state.get("vel_x",             0),
            state.get("vel_y",             0),
            state.get("vel_z",             0),
            float(state.get("onground",  True)),
            state.get("num_enemies",       0),
            float(state.get("blocked",     0)),
            # 8 raycasts (0 = wall right here, 1 = nothing within range)
            rays[0], rays[1], rays[2], rays[3],
            rays[4], rays[5], rays[6], rays[7],
        ], dtype=np.float32)

    def _compute_reward(self, state, action):
        reward = 0.0

        move_fb = float(action[0])
        shoot   = float(action[3])

        # frag reward / suicide penalty 
        current_frags = state.get("frags", 0)
        new_frags = current_frags - self.last_frags
        if new_frags > 0:
            reward += new_frags * 100.0
        elif new_frags < 0:
            reward -= 15.0
        self.last_frags = current_frags

        # death penalty 
        current_deaths = state.get("deaths", 0)
        new_deaths = max(0, current_deaths - self.last_deaths)
        reward -= new_deaths * 5.0
        self.last_deaths = current_deaths

        # distance tracking
        current_dist  = state.get("enemy_dist", 999.0)
        enemy_visible = state.get("enemy_visible", 0)
        moving_closer = (
            self.last_dist < 999.0 and
            (self.last_dist - current_dist) > 0 and
            enemy_visible
        )

        # aim reward (only when moving closer)
        enemy_angle = abs(state.get("enemy_angle_diff", 180.0))
        if moving_closer:
            aim_reward = max(0.0, (90.0 - enemy_angle) / 90.0) * 0.3
            reward += aim_reward
            if enemy_angle < 10.0:
                reward += 0.5    # precise aim bonus

        # distance reward
        if moving_closer:
            reward += 0.05
        self.last_dist = current_dist

        # visibility reward
        if enemy_visible:
            reward += 0.02  
            if current_dist < 50:
                reward += 0.05  
            elif current_dist < 100:
                reward += 0.02  

        # penalty for losing sight of enemy
        if not enemy_visible and self.last_enemy_visible:
            reward -= 0.05
        self.last_enemy_visible = enemy_visible

        # shooting discipline 
        if shoot >= 0.5:
            if not enemy_visible:
                reward -= 1.0          
            elif enemy_angle > 45.0:
                reward -= 0.5          
            elif enemy_angle > 20.0:
                reward -= 0.1          

        # movement rewards
        vel_mag = (state.get("vel_x", 0)**2 + state.get("vel_y", 0)**2) ** 0.5
        if vel_mag > 0.5:
            reward += 0.02             

        if move_fb > 0.5:
            reward += 0.05             
        elif move_fb < -0.5:
            reward -= 0.05             

        # wall collision penalties
        if move_fb > 0.5 and vel_mag < 0.5:
            reward -= 0.1              

        if state.get("blocked", 0):
            reward -= 0.2

        # raycast wall avoidance
        ray_front = state.get("ray0", 1.0)   
        if ray_front < 0.1:
            reward -= 0.2              

        # time penalty
        reward -= 0.01

        return reward

    def get_episode_stats(self):
        return self.episode_stats

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()