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
