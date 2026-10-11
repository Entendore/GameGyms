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
