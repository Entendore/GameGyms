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
