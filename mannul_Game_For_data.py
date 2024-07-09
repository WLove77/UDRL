import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.distributions import Categorical
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from minigrid.wrappers import RGBImgObsWrapper
from tqdm import tqdm
import math
import pygame
import pickle
from env import FourRoomsEnv

# 包装环境的函数
def create_wrapped_env(render_mode=None):
    env = FourRoomsEnv(render_mode=render_mode)
    env = RGBImgObsWrapper(env)
    return env

# 定义手动玩游戏并记录数据的函数
def manual_play_and_record(env, filename=" ", n_episodes=5):
    pygame.init()

    action_mapping = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2
    }

    action_names = {
        0: "left",
        1: "right",
        2: "up"
    }

    init_desired_reward = 1
    init_time_horizon = 1

    recorded_data = []

    if os.path.exists("recorded_episodes_lastest.pkl"):
        with open('recorded_episodes_lastest.pkl', 'rb') as f:
            recorded_data = pickle.load(f)

    for episode in range(n_episodes):
        print(f"Starting episode {episode + 1}")
        states, actions, rewards = [], [], []

        desired_return = torch.FloatTensor([init_desired_reward])
        desired_time_horizon = torch.FloatTensor([init_time_horizon])

        obs, _ = env.reset()
        state = obs['image']

        done, truncated = False, False
        step = 0
        total_reward = 0

        while not (done or truncated):
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in action_mapping:
                        action = action_mapping[event.key]
                        next_obs, reward, done, truncated, _ = env.step(action)
                        next_state = next_obs['image']

                        state_tensor = torch.from_numpy(state).float().permute(2, 0, 1)

                        states.append(state_tensor.cpu().numpy())
                        actions.append(action)
                        rewards.append(reward)

                        total_reward += reward
                        step += 1
                        print(f"Step {step}, reward={reward:.2f}")
                        print(f"Pressed {action_names[action]}")

                        state = next_state
                        desired_return -= reward
                        desired_time_horizon -= 1
                        desired_time_horizon = torch.FloatTensor([max(desired_time_horizon.item(), 1)])

                        if done or truncated:
                            if done:
                                print("Terminated!")
                            break
        print(f"Total Reward: {total_reward:.2f}")
        recorded_data.append((states, actions, rewards))

    with open(filename, 'wb') as f:
        pickle.dump(recorded_data, f)

    pygame.quit()

# Create the env
env = create_wrapped_env(render_mode='human')
action_space = 3
obs_space = env.observation_space['image'].shape
replay_size = 10000

print(f"Observation space: {obs_space}")
print(f"Action space: {action_space}")
print(f"Device: {device}")

# Record the data
manual_play_and_record(env, filename="recorded_episodes_lastest.pkl", n_episodes=50)
