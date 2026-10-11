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
