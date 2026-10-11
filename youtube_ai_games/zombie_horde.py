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

class ZombieHordeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 600
        self.height = 600
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-100, high=600, shape=(6,), dtype=np.float32)
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([300.0, 300.0])
        self.agent_hp = 100
        self.bullets = []
        self.zombies = []
        self.steps = 0
        self.max_steps = 600
        for _ in range(30):
            self._spawn_zombie()
        return self._get_obs(), {}

    def _spawn_zombie(self):
        x, y = random.uniform(0, self.width), random.uniform(0, self.height)
        if random.random() > 0.5:
            x = -20 if random.random() > 0.5 else self.width + 20
        else:
            y = -20 if random.random() > 0.5 else self.height + 20
        self.zombies.append([x, y])

    def _get_obs(self):
        if not self.zombies:
            return np.zeros(6, dtype=np.float32)
        z = np.array(self.zombies)
        dists = np.linalg.norm(z - self.agent_pos, axis=1)
        nearest = z[np.argmin(dists)]
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            nearest[0] - self.agent_pos[0], nearest[1] - self.agent_pos[1],
            self.agent_hp, len(self.zombies)
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        speed = 4.0

        if action == 0: self.agent_pos[1] -= speed
        elif action == 1: self.agent_pos[1] += speed
        elif action == 2: self.agent_pos[0] -= speed
        elif action == 3: self.agent_pos[0] += speed
        elif action == 4:
            obs = self._get_obs()
            if obs[2] != 0 or obs[3] != 0:
                angle = np.arctan2(obs[3], obs[2])
                self.bullets.append([self.agent_pos.copy(), angle, 8.0])

        for b in self.bullets:
            b[0][0] += np.cos(b[1]) * b[2]
            b[0][1] += np.sin(b[1]) * b[2]
        self.bullets = [b for b in self.bullets if 0 <= b[0][0] <= self.width and 0 <= b[0][1] <= self.height]

        new_zombies = []
        for z in self.zombies:
            z_pos = np.array(z)
            dist = np.linalg.norm(self.agent_pos - z_pos)
            if dist > 0:
                z_pos += (self.agent_pos - z_pos) / dist * 1.5
            hit = False
            for b in self.bullets:
                if np.linalg.norm(b[0] - z_pos) < 15:
                    reward += 1.0
                    hit = True
                    break
            if not hit:
                new_zombies.append(z_pos.tolist())
        self.zombies = new_zombies

        if self.steps % 20 == 0:
            for _ in range(5):
                self._spawn_zombie()

        for z in self.zombies:
            if np.linalg.norm(np.array(z) - self.agent_pos) < 20:
                self.agent_hp -= 2
                reward -= 0.5
                break

        if self.agent_hp <= 0:
            terminated = True
            reward -= 50.0
        reward += 0.05

        self.agent_pos[0] = np.clip(self.agent_pos[0], 10, self.width - 10)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 10, self.height - 10)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        self.screen.fill((20, 20, 20))
        for z in self.zombies:
            pygame.draw.circle(self.screen, (255, 50, 50), (int(z[0]), int(z[1])), 8)
        for b in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(b[0][0]), int(b[0][1])), 3)
        pygame.draw.circle(self.screen, (0, 150, 255), self.agent_pos.astype(int), 10)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

if __name__ == "__main__":
    GAME_NAME = "zombie_horde"
    env = ZombieHordeEnv()
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
