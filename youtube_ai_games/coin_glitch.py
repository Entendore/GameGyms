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
