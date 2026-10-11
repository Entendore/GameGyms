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
