import random
import math
import torch
import gymnasium as gym
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

class FourRoomsEnv(MiniGridEnv):
    def __init__(self, size=13, max_steps=None, **kwargs):
        self._rand = random.Random()
        self.goal_pos = None
        self.last_distance = None
        self.total_reward = 0
        self.data = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 500

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Find the terminal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, 1)
        self.grid.wall_rect(0, height - 1, width, 1)
        self.grid.wall_rect(0, 0, 1, height)
        self.grid.wall_rect(width - 1, 0, 1, height)

        self.grid.wall_rect(width // 2, 0, 1, height)
        self.grid.wall_rect(0, height // 2, width, 1)

        self.grid.set(width // 2, height // 4, None)
        self.grid.set(width // 2, 3 * height // 4, None)
        self.grid.set(width // 4, height // 2, None)
        self.grid.set(3 * width // 4, height // 2, None)

        self.goal_pos = self._get_random_position(width, height)
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        agent_pos = self._get_random_position(width, height)
        self.agent_pos = agent_pos
        self.agent_dir = self._rand_int(0, 3)

        self.mission = self._gen_mission()
        self.last_distance = self._compute_distance(self.agent_pos, self.goal_pos, method='manhattan')

        self.total_reward = 0

    def _get_random_position(self, width, height):
        while True:
            x = self._rand_int(1, width - 2)
            y = self._rand_int(1, height - 2)
            if self.grid.get(x, y) is None:
                return (x, y)

    def _rand_int(self, low, high):
        return self._rand.randint(low, high)

    def _compute_distance(self, pos1, pos2, method='manhattan'):
        if method == 'euclidean':
            return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        elif method == 'manhattan':
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        elif method == 'chebyshev':
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
        else:
            raise ValueError("Unknown distance method")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        current_distance = self._compute_distance(self.agent_pos, self.goal_pos, method='manhattan')

        if self.agent_pos == self.goal_pos:
            reward = 10
            terminated = True
        else:
            if self.last_distance is not None:
                if current_distance < self.last_distance:
                    reward += 1
                elif current_distance > self.last_distance:
                    reward -= 1
            reward -= 0.1

        self.total_reward += reward

        if terminated or truncated:
            print(f"Total Reward: {self.total_reward:.2f}")

        self.data.append({
            'state': obs,
            'action': action,
            'reward': reward,
            'next_state': self.gen_obs(),
            'done': terminated or truncated
        })

        self.last_distance = current_distance

        return obs, reward, terminated, truncated, info

    def get_data(self):
        return self.data