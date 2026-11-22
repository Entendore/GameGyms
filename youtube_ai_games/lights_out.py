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
