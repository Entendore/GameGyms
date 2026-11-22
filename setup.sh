#!/bin/bash
# ==============================================================================
# SCRIPT TO CREATE 12 AI GAME FILES (CLEAN FORMAT)
# ==============================================================================

DIR="youtube_ai_games"
mkdir -p "$DIR"

echo "Generating 12 cleanly formatted Python game files in the '$DIR' folder..."

cat << 'EOF' > "$DIR/nemesis_rts.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import random
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class NemesisRTSEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tile_size = 40
        self.grid_w = 20
        self.grid_h = 15
        self.width = self.grid_w * self.tile_size
        self.height = self.grid_h * self.tile_size
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.grid_w-1, self.grid_h-1, 150, self.grid_w-1, self.grid_h-1, 50, 50], dtype=np.float32)
        )
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terrain = np.zeros((self.grid_w, self.grid_h))
        for _ in range(15):
            x, y = random.randint(2, self.grid_w-3), random.randint(2, self.grid_h-3)
            self.terrain[x, y] = 1
            if random.random() > 0.5 and x + 1 < self.grid_w:
                self.terrain[x + 1, y] = 1
                
        self.agent_pos = np.array([2, 7])
        self.agent_hp = 150
        self.enemy_pos = np.array([17, 7])
        self.enemy_hp = 50
        self.steps = 0
        self.max_steps = 300
        return self._get_obs(), {}

    def _get_obs(self):
        distance = np.linalg.norm(self.agent_pos - self.enemy_pos)
        return np.array([
            self.agent_pos[0], self.agent_pos[1], float(self.agent_hp),
            self.enemy_pos[0], self.enemy_pos[1], float(self.enemy_hp),
            float(distance)
        ], dtype=np.float32)

    def _is_valid_move(self, x, y):
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return False
        return self.terrain[x, y] == 0

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        next_pos = self.agent_pos.copy()

        if action == 0: next_pos[1] -= 1
        elif action == 1: next_pos[1] += 1
        elif action == 2: next_pos[0] -= 1
        elif action == 3: next_pos[0] += 1
        elif action == 4:
            if np.linalg.norm(self.agent_pos - self.enemy_pos) <= 1.5:
                self.enemy_hp -= 40
                reward += 20.0

        if action < 4 and self._is_valid_move(next_pos[0], next_pos[1]):
            self.agent_pos = next_pos

        distance = np.linalg.norm(self.agent_pos - self.enemy_pos)
        if distance > 4:
            direction = (self.agent_pos - self.enemy_pos) / (distance + 1e-5)
            next_enemy_pos = self.enemy_pos + direction * 1.0
            if self._is_valid_move(int(next_enemy_pos[0]), int(next_enemy_pos[1])):
                self.enemy_pos = next_enemy_pos.astype(int)
        elif distance <= 2:
            direction = (self.enemy_pos - self.agent_pos) / (distance + 1e-5)
            next_enemy_pos = self.enemy_pos + direction * 1.5
            if self._is_valid_move(int(next_enemy_pos[0]), int(next_enemy_pos[1])):
                self.enemy_pos = next_enemy_pos.astype(int)

        if distance <= 5:
            self.agent_hp -= 15
            reward -= 5.0

        reward -= 0.1 * distance

        if self.enemy_hp <= 0:
            terminated = True
            reward += 200.0
        elif self.agent_hp <= 0:
            terminated = True
            reward -= 100.0

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                color = (80, 80, 80) if self.terrain[x, y] == 1 else (40, 60, 40)
                pygame.draw.rect(self.screen, color, (x*self.tile_size, y*self.tile_size, self.tile_size, self.tile_size))
        
        ax = int(self.agent_pos[0] * self.tile_size + 20)
        ay = int(self.agent_pos[1] * self.tile_size + 20)
        ex = int(self.enemy_pos[0] * self.tile_size + 20)
        ey = int(self.enemy_pos[1] * self.tile_size + 20)
        
        pygame.draw.circle(self.screen, (255, 50, 50), (ex, ey), 15)
        pygame.draw.rect(self.screen, (0, 100, 255), (ax-15, ay-15, 30, 30))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "nemesis_rts"
    env = NemesisRTSEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=60000, callback=CheckpointCallback(15000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=30)
EOF

cat << 'EOF' > "$DIR/gladiator_arena.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class GladiatorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 600
        self.height = 400
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=600, shape=(7,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([100.0, 200.0])
        self.agent_hp = 100
        self.enemy_pos = np.array([500.0, 200.0])
        self.enemy_hp = 100
        self.steps = 0
        self.max_steps = 500
        return self._get_obs(), {}

    def _get_obs(self):
        distance = np.linalg.norm(self.agent_pos - self.enemy_pos)
        return np.array([
            self.agent_pos[0], self.agent_pos[1], float(self.agent_hp),
            self.enemy_pos[0], self.enemy_pos[1], float(self.enemy_hp),
            float(distance)
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        speed = 5.0

        if action == 0: self.agent_pos[1] = max(15, self.agent_pos[1] - speed)
        elif action == 1: self.agent_pos[1] = min(self.height - 15, self.agent_pos[1] + speed)
        elif action == 2: self.agent_pos[0] = max(15, self.agent_pos[0] - speed)
        elif action == 3: self.agent_pos[0] = min(self.width - 15, self.agent_pos[0] + speed)
        elif action == 4:
            if np.linalg.norm(self.agent_pos - self.enemy_pos) <= 50:
                self.enemy_hp -= 10
                reward += 10.0

        distance = np.linalg.norm(self.agent_pos - self.enemy_pos)
        if distance > 30:
            direction = (self.agent_pos - self.enemy_pos) / (distance + 1e-5)
            self.enemy_pos += direction * 3.0
        else:
            self.agent_hp -= 15
            reward -= 5.0

        reward -= 0.01
        if self.enemy_hp <= 0:
            terminated = True
            reward += 100.0
        elif self.agent_hp <= 0:
            terminated = True
            reward -= 50.0

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((30, 30, 30))
        pygame.draw.circle(self.screen, (255, 50, 50), self.enemy_pos.astype(int), 15)
        pygame.draw.circle(self.screen, (0, 150, 255), self.agent_pos.astype(int), 15)
        pygame.draw.circle(self.screen, (50, 50, 50), self.agent_pos.astype(int), 50, 1)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "gladiator_arena"
    env = GladiatorEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=60000, callback=CheckpointCallback(15000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=30)
EOF

cat << 'EOF' > "$DIR/sokoban_puzzle.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class SokobanEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tile_size = 50
        self.grid_w = 8
        self.grid_h = 6
        self.width = self.grid_w * self.tile_size
        self.height = self.grid_h * self.tile_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=7, shape=(6,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([1, 1])
        self.box_pos = np.array([4, 2])
        self.target_pos = np.array([6, 4])
        self.steps = 0
        self.max_steps = 100
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.box_pos[0], self.box_pos[1],
            self.target_pos[0], self.target_pos[1]
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        next_pos = self.agent_pos.copy()

        if action == 0: next_pos[1] -= 1
        elif action == 1: next_pos[1] += 1
        elif action == 2: next_pos[0] -= 1
        elif action == 3: next_pos[0] += 1

        if 0 <= next_pos[0] < self.grid_w and 0 <= next_pos[1] < self.grid_h:
            if np.array_equal(next_pos, self.box_pos):
                next_box_pos = self.box_pos + (next_pos - self.agent_pos)
                if 0 <= next_box_pos[0] < self.grid_w and 0 <= next_box_pos[1] < self.grid_h:
                    self.box_pos = next_box_pos
                    self.agent_pos = next_pos
                    if np.linalg.norm(self.box_pos - self.target_pos) < np.linalg.norm(next_pos - self.target_pos):
                        reward += 0.1
            else:
                self.agent_pos = next_pos

        if np.array_equal(self.box_pos, self.target_pos):
            reward = 10.0
            terminated = True
        else:
            reward -= 0.01

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((50, 50, 50))
        tx, ty = int(self.target_pos[0] * self.tile_size), int(self.target_pos[1] * self.tile_size)
        pygame.draw.rect(self.screen, (0, 255, 0), (tx, ty, self.tile_size, self.tile_size))
        bx, by = int(self.box_pos[0] * self.tile_size), int(self.box_pos[1] * self.tile_size)
        pygame.draw.rect(self.screen, (255, 255, 0), (bx+5, by+5, self.tile_size-10, self.tile_size-10))
        ax, ay = int(self.agent_pos[0] * self.tile_size), int(self.agent_pos[1] * self.tile_size)
        pygame.draw.circle(self.screen, (0, 100, 255), (ax + self.tile_size//2, ay + self.tile_size//2), self.tile_size//2 - 5)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "sokoban_puzzle"
    env = SokobanEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=60000, callback=CheckpointCallback(15000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=10)
EOF

cat << 'EOF' > "$DIR/zombie_horde.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import random
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class ZombieHordeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 600
        self.height = 600
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-100, high=600, shape=(6,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([300.0, 300.0])
        self.agent_hp = 100
        self.bullets = []
        self.zombies = []
        self.steps = 0
        self.max_steps = 600
        for _ in range(30):
            self._spawn_zombie()
        return self._get_obs(), {}

    def _spawn_zombie(self):
        x, y = random.uniform(0, self.width), random.uniform(0, self.height)
        if random.random() > 0.5:
            x = -20 if random.random() > 0.5 else self.width + 20
        else:
            y = -20 if random.random() > 0.5 else self.height + 20
        self.zombies.append([x, y])

    def _get_obs(self):
        if not self.zombies:
            return np.zeros(6, dtype=np.float32)
        z = np.array(self.zombies)
        dists = np.linalg.norm(z - self.agent_pos, axis=1)
        nearest = z[np.argmin(dists)]
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            nearest[0] - self.agent_pos[0], nearest[1] - self.agent_pos[1],
            self.agent_hp, len(self.zombies)
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        speed = 4.0

        if action == 0: self.agent_pos[1] -= speed
        elif action == 1: self.agent_pos[1] += speed
        elif action == 2: self.agent_pos[0] -= speed
        elif action == 3: self.agent_pos[0] += speed
        elif action == 4:
            obs = self._get_obs()
            if obs[2] != 0 or obs[3] != 0:
                angle = np.arctan2(obs[3], obs[2])
                self.bullets.append([self.agent_pos.copy(), angle, 8.0])

        for b in self.bullets:
            b[0][0] += np.cos(b[1]) * b[2]
            b[0][1] += np.sin(b[1]) * b[2]
        self.bullets = [b for b in self.bullets if 0 <= b[0][0] <= self.width and 0 <= b[0][1] <= self.height]

        new_zombies = []
        for z in self.zombies:
            z_pos = np.array(z)
            dist = np.linalg.norm(self.agent_pos - z_pos)
            if dist > 0:
                z_pos += (self.agent_pos - z_pos) / dist * 1.5
            hit = False
            for b in self.bullets:
                if np.linalg.norm(b[0] - z_pos) < 15:
                    reward += 1.0
                    hit = True
                    break
            if not hit:
                new_zombies.append(z_pos.tolist())
        self.zombies = new_zombies

        if self.steps % 20 == 0:
            for _ in range(5):
                self._spawn_zombie()

        for z in self.zombies:
            if np.linalg.norm(np.array(z) - self.agent_pos) < 20:
                self.agent_hp -= 2
                reward -= 0.5
                break

        if self.agent_hp <= 0:
            terminated = True
            reward -= 50.0
        reward += 0.05

        self.agent_pos[0] = np.clip(self.agent_pos[0], 10, self.width - 10)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 10, self.height - 10)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((20, 20, 20))
        for z in self.zombies:
            pygame.draw.circle(self.screen, (255, 50, 50), (int(z[0]), int(z[1])), 8)
        for b in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(b[0][0]), int(b[0][1])), 3)
        pygame.draw.circle(self.screen, (0, 150, 255), self.agent_pos.astype(int), 10)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "zombie_horde"
    env = ZombieHordeEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=60000, callback=CheckpointCallback(15000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=30)
EOF

cat << 'EOF' > "$DIR/coin_glitch.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class CoinGlitchEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 600
        self.height = 400
        self.ground_y = 350
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-100, high=600, shape=(4,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([50.0, float(self.ground_y)])
        self.vel_y = 0.0
        self.steps = 0
        self.max_steps = 300
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], 400.0, self.ground_y], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        self.vel_y += 0.5
        self.agent_pos[1] += self.vel_y
        
        if self.agent_pos[1] >= self.ground_y:
            self.agent_pos[1] = self.ground_y
            self.vel_y = 0.0
            if action == 2:
                self.vel_y = -10.0
                
        if action == 0: self.agent_pos[0] -= 3.0
        elif action == 1: self.agent_pos[0] += 3.0
        
        if 140 <= self.agent_pos[0] <= 160 and self.vel_y < 0 and self.agent_pos[1] < 300:
            reward += 1.0
            
        if self.agent_pos[0] >= 500:
            reward = 50.0
            terminated = True
            
        reward -= 0.01
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((135, 206, 235))
        pygame.draw.rect(self.screen, (100, 70, 50), (0, self.ground_y, self.width, 50))
        pygame.draw.rect(self.screen, (255, 200, 0), (140, 200, 20, 50))
        pygame.draw.rect(self.screen, (0, 255, 0), (500, self.ground_y - 80, 10, 80))
        pygame.draw.circle(self.screen, (255, 0, 0), self.agent_pos.astype(int), 15)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "coin_glitch"
    env = CoinGlitchEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50000, callback=CheckpointCallback(12500, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=30)
EOF

cat << 'EOF' > "$DIR/bullet_hell.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import math
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class BulletHellEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 400
        self.height = 400
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-400, high=400, shape=(6,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([200.0, 200.0])
        self.bullets = []
        self.steps = 0
        self.max_steps = 400
        self.spawn_timer = 0
        return self._get_obs(), {}

    def _get_obs(self):
        if not self.bullets:
            return np.zeros(6, dtype=np.float32)
        b = np.array(self.bullets)
        dists = np.linalg.norm(b[:, :2] - self.agent_pos, axis=1)
        nearest = b[np.argmin(dists)]
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            nearest[0] - self.agent_pos[0], nearest[1] - self.agent_pos[1],
            nearest[2], nearest[3]
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        speed = 3.0

        if action == 0: self.agent_pos[1] -= speed
        elif action == 1: self.agent_pos[1] += speed
        elif action == 2: self.agent_pos[0] -= speed
        elif action == 3: self.agent_pos[0] += speed

        self.agent_pos[0] = np.clip(self.agent_pos[0], 10, self.width - 10)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 10, self.height - 10)

        self.spawn_timer += 1
        if self.spawn_timer % 2 == 0:
            angle = self.steps * 0.1
            for i in range(4):
                a2 = angle + (i * math.pi / 2)
                vx, vy = math.cos(a2) * 2.0, math.sin(a2) * 2.0
                self.bullets.append([200.0, 200.0, vx, vy])

        for b in self.bullets:
            b[0] += b[2]
            b[1] += b[3]
        self.bullets = [b for b in self.bullets if -50 <= b[0] <= 450 and -50 <= b[1] <= 450]

        for b in self.bullets:
            if np.linalg.norm(np.array([b[0], b[1]]) - self.agent_pos) < 10:
                terminated = True
                reward = -50.0
                break
        if not terminated:
            reward += 0.1
            
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        for b in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(b[0]), int(b[1])), 3)
        pygame.draw.circle(self.screen, (0, 150, 255), self.agent_pos.astype(int), 8)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "bullet_hell"
    env = BulletHellEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50000, callback=CheckpointCallback(12500, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=30)
EOF

cat << 'EOF' > "$DIR/ice_slider.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import random
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class IceSliderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tile_size = 50
        self.grid_w = 10
        self.grid_h = 10
        self.width = self.grid_w * self.tile_size
        self.height = self.grid_h * self.tile_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([1, 1])
        self.target_pos = np.array([8, 8])
        self.trap_pos = np.array([5, 5])
        self.walls = []
        for _ in range(15):
            wx, wy = random.randint(0, 9), random.randint(0, 9)
            if not np.array_equal([wx, wy], self.agent_pos) and not np.array_equal([wx, wy], self.target_pos):
                self.walls.append((wx, wy))
        self.steps = 0
        self.max_steps = 50
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], self.target_pos[0], self.target_pos[1]], dtype=np.float32)

    def _is_blocked(self, x, y):
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return True
        return (x, y) in self.walls

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        dx, dy = dirs[action]
        next_pos = self.agent_pos.copy()

        while True:
            nx, ny = next_pos[0] + dx, next_pos[1] + dy
            if self._is_blocked(nx, ny):
                break
            next_pos = [nx, ny]
            if np.array_equal(next_pos, self.trap_pos):
                terminated = True
                reward = -10.0
                break

        self.agent_pos = np.array(next_pos)
        if not terminated and np.array_equal(self.agent_pos, self.target_pos):
            terminated = True
            reward = 20.0
        reward -= 0.1
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((50, 50, 50))
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if (x, y) in self.walls:
                    pygame.draw.rect(self.screen, (100, 100, 100), (x*self.tile_size, y*self.tile_size, self.tile_size, self.tile_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.trap_pos[0]*self.tile_size, self.trap_pos[1]*self.tile_size, self.tile_size, self.tile_size))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.target_pos[0]*self.tile_size, self.target_pos[1]*self.tile_size, self.tile_size, self.tile_size))
        pygame.draw.circle(self.screen, (0, 150, 255), (self.agent_pos[0]*self.tile_size + self.tile_size//2, self.agent_pos[1]*self.tile_size + self.tile_size//2), 15)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "ice_slider"
    env = IceSliderEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50000, callback=CheckpointCallback(12500, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=15)
EOF

cat << 'EOF' > "$DIR/neural_snake.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import random
from collections import deque
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class SnakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tile_size = 30
        self.grid_w = 15
        self.grid_h = 15
        self.width = self.grid_w * self.tile_size
        self.height = self.grid_h * self.tile_size
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-15, high=15, shape=(5,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.body = deque([(7, 7), (6, 7), (5, 7)])
        self.dir = (1, 0)
        self._spawn_apple()
        self.steps = 0
        self.max_steps = 200
        return self._get_obs(), {}

    def _spawn_apple(self):
        while True:
            self.apple = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if self.apple not in self.body:
                break

    def _get_obs(self):
        hx, hy = self.body[0]
        ax, ay = self.apple[0] - hx, self.apple[1] - hy
        df = (hx + self.dir[0], hy + self.dir[1])
        ld = (self.dir[1], -self.dir[0])
        dl = (hx + ld[0], hy + ld[1])
        rd = (-self.dir[1], self.dir[0])
        dr = (hx + rd[0], hy + rd[1])
        d = lambda p: p[0] < 0 or p[0] >= self.grid_w or p[1] < 0 or p[1] >= self.grid_h or p in self.body
        return np.array([ax, ay, d(df), d(dl), d(dr)], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        if action == 1:
            self.dir = (self.dir[1], -self.dir[0])
        elif action == 2:
            self.dir = (-self.dir[1], self.dir[0])
            
        new_head = (self.body[0][0] + self.dir[0], self.body[0][1] + self.dir[1])
        if new_head[0] < 0 or new_head[0] >= self.grid_w or new_head[1] < 0 or new_head[1] >= self.grid_h or new_head in self.body:
            terminated = True
            reward = -10.0
        else:
            self.body.appendleft(new_head)
            if new_head == self.apple:
                reward = 10.0
                self._spawn_apple()
            else:
                self.body.pop()
                
        reward -= 0.01
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((20, 20, 20))
        pygame.draw.rect(self.screen, (255, 50, 50), (self.apple[0]*self.tile_size, self.apple[1]*self.tile_size, self.tile_size, self.tile_size))
        for i, s in enumerate(self.body):
            c = (0, 255, 0) if i == 0 else (0, 150, 0)
            pygame.draw.rect(self.screen, c, (s[0]*self.tile_size, s[1]*self.tile_size, self.tile_size, self.tile_size))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "neural_snake"
    env = SnakeEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=80000, callback=CheckpointCallback(20000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=15)
EOF

cat << 'EOF' > "$DIR/lights_out.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import random
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class LightsOutEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 4
        self.tile_size = 80
        self.width = self.grid_size * self.tile_size
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.width))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.ones(16, dtype=np.float32)
        for _ in range(5):
            self._toggle(random.randint(0, 15))
        self.steps = 0
        self.max_steps = 50
        return self._get_obs(), {}

    def _get_obs(self):
        return self.grid.copy()

    def _toggle(self, i):
        x, y = i % self.grid_size, i // self.grid_size
        self.grid[i] = 1.0 - self.grid[i]
        if x > 0: self.grid[y * self.grid_size + (x - 1)] = 1.0 - self.grid[y * self.grid_size + (x - 1)]
        if x < self.grid_size - 1: self.grid[y * self.grid_size + (x + 1)] = 1.0 - self.grid[y * self.grid_size + (x + 1)]
        if y > 0: self.grid[(y - 1) * self.grid_size + x] = 1.0 - self.grid[(y - 1) * self.grid_size + x]
        if y < self.grid_size - 1: self.grid[(y + 1) * self.grid_size + x] = 1.0 - self.grid[(y + 1) * self.grid_size + x]

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        self._toggle(action)
        on = np.sum(self.grid)
        reward = -on * 0.1
        if on == 0:
            terminated = True
            reward = 50.0
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((30, 30, 30))
        for i in range(16):
            x = (i % self.grid_size) * self.tile_size
            y = (i // self.grid_size) * self.tile_size
            color = (255, 255, 0) if self.grid[i] == 1 else (50, 50, 50)
            pygame.draw.rect(self.screen, color, (x + 5, y + 5, self.tile_size - 10, self.tile_size - 10))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "lights_out"
    env = LightsOutEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=60000, callback=CheckpointCallback(15000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=10)
EOF

cat << 'EOF' > "$DIR/hanoi.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class HanoiEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 400
        self.height = 300
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pegs = [[3, 2, 1], [], []]
        self.steps = 0
        self.max_steps = 50
        return self._get_obs(), {}

    def _get_obs(self):
        dp = [0, 0, 0]
        for pi, p in enumerate(self.pegs):
            for d in p:
                dp[d - 1] = pi
        return np.array(dp, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -0.01
        terminated = False
        moves = {0: [0, 1], 1: [0, 2], 2: [1, 0], 3: [1, 2], 4: [2, 0], 5: [2, 1]}
        s, d = moves[action]
        if self.pegs[s]:
            dk = self.pegs[s][-1]
            if not self.pegs[d] or self.pegs[d][-1] > dk:
                self.pegs[s].pop()
                self.pegs[d].append(dk)
                if dk == 1 and d == 2: reward += 0.5
                if dk == 2 and d == 2 and 1 in self.pegs[2]: reward += 1.0
            else:
                reward -= 0.5
        else:
            reward -= 0.5
        if self.pegs[2] == [3, 2, 1]:
            terminated = True
            reward = 20.0
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((30, 30, 30))
        colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 100, 255)}
        sizes = {1: 30, 2: 50, 3: 70}
        for pi, p in enumerate(self.pegs):
            xc = 60 + pi * 140
            pygame.draw.rect(self.screen, (100, 100, 100), (xc - 5, 50, 10, 200))
            pygame.draw.rect(self.screen, (150, 150, 150), (xc - 40, 250, 80, 10))
            for di, dk in enumerate(p):
                w = sizes[dk]
                y = 240 - (di * 25)
                pygame.draw.rect(self.screen, colors[dk], (xc - w//2, y, w, 20))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "hanoi"
    env = HanoiEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=40000, callback=CheckpointCallback(10000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=5)
EOF

cat << 'EOF' > "$DIR/rolling_maze.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class RollingMazeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 400
        self.height = 400
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-400, high=400, shape=(6,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([50.0, 50.0])
        self.vel = np.array([0.0, 0.0])
        self.target = np.array([350.0, 350.0])
        self.steps = 0
        self.max_steps = 200
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.target[0], self.target[1]], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        g = 0.5
        
        if action == 1: self.vel[0] -= g
        elif action == 2: self.vel[0] += g
        elif action == 3: self.vel[1] -= g
        elif action == 4: self.vel[1] += g
        
        self.vel *= 0.9
        self.pos += self.vel
        
        if self.pos[0] < 10 or self.pos[0] > self.width - 10 or self.pos[1] < 10 or self.pos[1] > self.height - 10:
            terminated = True
            reward = -20.0
        else:
            dist = np.linalg.norm(self.pos - self.target)
            reward = -0.1 * dist
            if dist < 20:
                terminated = True
                reward = 50.0
                
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((20, 20, 20))
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, self.width - 20, self.height - 20))
        pygame.draw.circle(self.screen, (0, 0, 0), self.target.astype(int), 15)
        pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), 8)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "rolling_maze"
    env = RollingMazeEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=50000, callback=CheckpointCallback(12500, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=30)
EOF

cat << 'EOF' > "$DIR/8_puzzle.py"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import random
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class EightPuzzleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tile_size = 80
        self.grid_size = 3
        self.width = self.tile_size * self.grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=8, shape=(9,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.width))
        self.font = pygame.font.Font(None, 40)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tiles = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])
        for _ in range(20):
            self._move(random.randint(0, 3))
        self.steps = 0
        self.max_steps = 50
        return self._get_obs(), {}

    def _get_obs(self):
        return self.tiles.astype(np.float32)

    def _move(self, action):
        blank_idx = np.where(self.tiles == 0)[0][0]
        x, y = blank_idx % 3, blank_idx // 3
        if action == 0 and y > 0:
            self.tiles[blank_idx], self.tiles[blank_idx - 3] = self.tiles[blank_idx - 3], self.tiles[blank_idx]
        elif action == 1 and y < 2:
            self.tiles[blank_idx], self.tiles[blank_idx + 3] = self.tiles[blank_idx + 3], self.tiles[blank_idx]
        elif action == 2 and x > 0:
            self.tiles[blank_idx], self.tiles[blank_idx - 1] = self.tiles[blank_idx - 1], self.tiles[blank_idx]
        elif action == 3 and x < 2:
            self.tiles[blank_idx], self.tiles[blank_idx + 1] = self.tiles[blank_idx + 1], self.tiles[blank_idx]

    def step(self, action):
        self.steps += 1
        reward = -0.01
        terminated = False
        self._move(action)
        dist = 0
        for i, t in enumerate(self.tiles):
            if t != 0:
                cx, cy = i % 3, i // 3
                tx, ty = (t - 1) % 3, (t - 1) // 3
                dist += abs(cx - tx) + abs(cy - ty)
        reward -= dist * 0.1
        if np.array_equal(self.tiles, np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])):
            terminated = True
            reward = 50.0
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((30, 30, 30))
        for i, t in enumerate(self.tiles):
            x = (i % 3) * self.tile_size
            y = (i // 3) * self.tile_size
            if t != 0:
                pygame.draw.rect(self.screen, (0, 100, 200), (x + 2, y + 2, self.tile_size - 4, self.tile_size - 4))
                tx = self.font.render(str(t), True, (255, 255, 255))
                self.screen.blit(tx, (x + self.tile_size//2 - 10, y + self.tile_size//2 - 10))
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "8_puzzle"
    env = EightPuzzleEnv()
    os.makedirs(f"{GAME_NAME}/checkpoints", exist_ok=True)
    os.makedirs(f"{GAME_NAME}/videos", exist_ok=True)
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=60000, callback=CheckpointCallback(15000, f"{GAME_NAME}/checkpoints", "model"))
    model.save(f"{GAME_NAME}/checkpoints/final.zip")
    
    for file_path in sorted(glob.glob(f"{GAME_NAME}/checkpoints/*.zip")):
        loaded_model = PPO.load(file_path, env=env)
        obs, _ = env.reset()
        frames = []
        done = False
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc: done = True
        video_name = file_path.split('/')[-1].replace('.zip', '.mp4')
        imageio.mimsave(f"{GAME_NAME}/videos/{video_name}", frames, fps=5)
EOF

echo "All 12 cleanly formatted Python game files created successfully in the '$DIR' folder."
echo "You can now navigate to the folder and run them manually whenever you are ready."