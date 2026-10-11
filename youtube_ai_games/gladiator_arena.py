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
