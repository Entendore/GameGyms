import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import gymnasium as gym
import imageio
import glob
import math
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class BulletHellEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 400
        self.height = 400
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-400, high=400, shape=(6,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([200.0, 200.0])
        self.bullets = []
        self.steps = 0
        self.max_steps = 400
        self.spawn_timer = 0
        return self._get_obs(), {}

    def _get_obs(self):
        if not self.bullets:
            return np.zeros(6, dtype=np.float32)
        b = np.array(self.bullets)
        dists = np.linalg.norm(b[:, :2] - self.agent_pos, axis=1)
        nearest = b[np.argmin(dists)]
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            nearest[0] - self.agent_pos[0], nearest[1] - self.agent_pos[1],
            nearest[2], nearest[3]
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        speed = 3.0

        if action == 0: self.agent_pos[1] -= speed
        elif action == 1: self.agent_pos[1] += speed
        elif action == 2: self.agent_pos[0] -= speed
        elif action == 3: self.agent_pos[0] += speed

        self.agent_pos[0] = np.clip(self.agent_pos[0], 10, self.width - 10)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 10, self.height - 10)

        self.spawn_timer += 1
        if self.spawn_timer % 2 == 0:
            angle = self.steps * 0.1
            for i in range(4):
                a2 = angle + (i * math.pi / 2)
                vx, vy = math.cos(a2) * 2.0, math.sin(a2) * 2.0
                self.bullets.append([200.0, 200.0, vx, vy])

        for b in self.bullets:
            b[0] += b[2]
            b[1] += b[3]
        self.bullets = [b for b in self.bullets if -50 <= b[0] <= 450 and -50 <= b[1] <= 450]

        for b in self.bullets:
            if np.linalg.norm(np.array([b[0], b[1]]) - self.agent_pos) < 10:
                terminated = True
                reward = -50.0
                break
        if not terminated:
            reward += 0.1
            
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        for b in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(b[0]), int(b[1])), 3)
        pygame.draw.circle(self.screen, (0, 150, 255), self.agent_pos.astype(int), 8)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "bullet_hell"
    env = BulletHellEnv()
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
