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
