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
