import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import warnings
from torch import nn
from minigrid.wrappers import RGBImgObsWrapper
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_sample(self, states, actions, rewards):
        episode = {"states": states, "actions": actions, "rewards": rewards, "summed_rewards": sum(rewards)}
        self.buffer.append(episode)
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda i: i["summed_rewards"], reverse=True)
            self.buffer = self.buffer[:self.max_size]

    def get_random_samples(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("Batch size larger than buffer size.")
        return random.sample(self.buffer, batch_size)

    def get_nbest(self, n):
        if n > len(self.buffer):
            raise ValueError("n is larger than buffer size.")
        self.buffer.sort(key=lambda i: i["summed_rewards"], reverse=True)
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        print("Buffer has been cleared.")


class BF(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size, seed):
        super(BF, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        conv_output_size = self._get_conv_output_size(obs_space)

        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.commands = nn.Linear(2, hidden_size)
        self.fc_comb = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def _get_conv_output_size(self, obs_space):
        with torch.no_grad():
            input = torch.zeros(1, obs_space[2], obs_space[0], obs_space[1])
            output = self.conv1(input)
            output = self.relu(output)
            output = self.conv2(output)
            output = self.relu(output)
            output = self.pool(output)
            output = output.view(1, -1)
        return output.size(1)

    def forward(self, state, command):
        state = self.relu(self.conv1(state))
        state = self.relu(self.conv2(state))
        state = self.pool(state)
        state = state.view(state.size(0), -1)

        state_out = self.relu(self.fc1(state))
        command_out = self.relu(self.commands(command))

        combined = torch.cat((state_out, command_out), dim=1)
        combined = self.relu(self.fc_comb(combined))
        combined = self.relu(self.fc2(combined))
        action_probs = self.fc3(combined)
        return action_probs

    def action(self, state, desire, horizon, return_scale, horizon_scale):
        command = torch.cat((desire * return_scale, horizon * horizon_scale), dim=-1).unsqueeze(0)
        action_prob = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action

    def greedy_action(self, state, desire, horizon, return_scale, horizon_scale):
        command = torch.cat((desire * return_scale, horizon * horizon_scale), dim=-1).unsqueeze(0)
        action_prob = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        action = torch.argmax(probs).item()
        return action


class UDRL:
    def __init__(self, env, buffer_size, hidden_size, learning_rate, return_scale, horizon_scale, batch_size,
                 n_updates_per_iter, n_episodes_per_iter, last_few, mode='auto', filename='', n_warm_up_episodes=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filename = filename
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.obs_space = env.observation_space['image'].shape
        self.action_space = env.action_space.n
        self.bf = BF(self.obs_space, self.action_space, hidden_size, seed=1).to(self.device)
        self.optimizer = optim.Adam(params=self.bf.parameters(), lr=learning_rate)
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        self.batch_size = batch_size
        self.n_updates_per_iter = n_updates_per_iter
        self.n_episodes_per_iter = n_episodes_per_iter
        self.last_few = last_few
        self.mode = mode
        self.n_warm_up_episodes = n_warm_up_episodes

        if mode == 'auto':
            self.warm_up()
        elif mode == 'manual':
            if filename:
                self.load_data_to_replay_buffer(filename)
            else:
                warnings.warn('FilenameError: Choose correct filename')
        else:
            warnings.warn("ModeError: Choose mode 'auto' or 'manual'")

    def load_data_to_replay_buffer(self, filename="recorded_episodes.pkl"):
        with open(filename, 'rb') as f:
            recorded_data = pickle.load(f)

        for episode in recorded_data:
            states, actions, rewards = episode
            self.buffer.add_sample(states, actions, rewards)

    def warm_up(self):
        init_desired_reward = 1
        init_time_horizon = 1
        for i in range(self.n_warm_up_episodes):
            desired_return = torch.FloatTensor([init_desired_reward])
            desired_time_horizon = torch.FloatTensor([init_time_horizon])
            obs, _ = self.env.reset()
            state = obs['image']
            states, actions, rewards = [], [], []
            step, total_reward = 0, 0
            while True:
                state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).to(self.device)
                action = self.bf.action(state_tensor.unsqueeze(0).to(self.device), desired_return.to(self.device),
                                        desired_time_horizon.to(self.device), self.return_scale, self.horizon_scale).item()
                next_obs, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                next_state = next_obs['image']

                step += 1

                states.append(state_tensor.cpu().numpy())
                actions.append(action)
                rewards.append(reward)

                state = next_state
                desired_return -= reward
                desired_time_horizon -= 1
                desired_time_horizon = torch.FloatTensor([max(desired_time_horizon.item(), 1)])
                if done or truncated:
                    logging.info(
                        f"Step {step}: Action = {action}, TotalReward = {total_reward:.2f}, Done = {done}, Truncated = {truncated}",flush=True)
                    break
            self.buffer.add_sample(states, actions, rewards)

    def sampling_exploration(self):
        if len(self.buffer) < self.last_few:
            raise ValueError("Not enough episodes in the buffer to sample from.")
        top_X = self.buffer.get_nbest(self.last_few)
        new_desired_horizon = np.mean([len(i["states"]) for i in top_X])
        returns = [i["summed_rewards"] for i in top_X]
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        new_desired_reward = np.random.uniform(mean_returns, mean_returns + std_returns)
        return torch.FloatTensor([new_desired_reward]), torch.FloatTensor([new_desired_horizon])

    def select_time_steps(self, saved_episode):
        T = len(saved_episode["states"])
        t1 = np.random.randint(0, T - 1)
        t2 = np.random.randint(t1 + 1, T)
        return t1, t2, T

    def create_training_input(self, episode, t1, t2):
        state = episode["states"][t1]
        desired_reward = sum(episode["rewards"][t1:t2])
        time_horizon = t2 - t1
        action = episode["actions"][t1]
        return state, desired_reward, time_horizon, action

    def create_training_examples(self, batch_size):
        states, rewards, horizons, actions = [], [], [], []
        episodes = self.buffer.get_random_samples(batch_size)
        for ep in episodes:
            t1, t2, T = self.select_time_steps(ep)
            state, desired_reward, time_horizon, action = self.create_training_input(ep, t1, t2)
            states.append(torch.FloatTensor(state))
            rewards.append(torch.FloatTensor([desired_reward]))
            horizons.append(torch.FloatTensor([time_horizon]))
            actions.append(torch.tensor(action, dtype=torch.long))
        return states, rewards, horizons, actions

    def train_behavior_function(self, batch_size):
        states, rewards, horizons, actions = self.create_training_examples(batch_size)
        state_tensors = torch.stack(states).to(self.device)
        reward_tensors = torch.stack(rewards).to(self.device)
        horizon_tensors = torch.stack(horizons).to(self.device)
        action_tensors = torch.stack(actions).to(self.device)
        command = torch.cat((reward_tensors, horizon_tensors), dim=1)
        outputs = self.bf(state_tensors, command).float()
        loss = F.cross_entropy(outputs, action_tensors)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, desired_return, desired_time_horizon):
        obs, _ = self.env.reset()
        state = obs['image']
        total_rewards = 0
        while True:
            state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            action = self.bf.action(state_tensor, desired_return.to(self.device),
                                    desired_time_horizon.to(self.device), self.return_scale, self.horizon_scale).item()
            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_state = next_obs['image']
            total_rewards += reward
            state = next_state
            desired_return -= reward
            desired_time_horizon -= 1
            desired_time_horizon = torch.FloatTensor([max(desired_time_horizon.item(), 1)])
            if done or truncated:
                break
        return total_rewards

    def generate_episode(self, desired_return=torch.FloatTensor([1]), desired_time_horizon=torch.FloatTensor([1])):
        obs, _ = self.env.reset()
        state = obs['image']
        states, actions, rewards = [], [], []
        while True:
            state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).to(self.device)
            action = self.bf.action(state_tensor.unsqueeze(0).to(self.device), desired_return.to(self.device),
                                    desired_time_horizon.to(self.device), self.return_scale, self.horizon_scale).item()
            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_state = next_obs['image']
            states.append(state_tensor.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            state = next_state
            desired_return -= reward
            desired_time_horizon -= 1
            desired_time_horizon = torch.FloatTensor([np.maximum(desired_time_horizon, 1).item()])
            if done:
                break
        if len(states) < 2:
            return self.generate_episode(desired_return, desired_time_horizon)
        return [states, actions, rewards]

    def run_upside_down(self, max_episodes):
        all_rewards, losses, average_100_reward, desired_rewards_history, horizon_history = [], [], [], [], []
        for ep in tqdm(range(1, max_episodes + 1), desc="Training Progress"):
            loss_buffer = [self.train_behavior_function(self.batch_size) for _ in range(self.n_updates_per_iter)]
            losses.append(np.mean(loss_buffer))
            for _ in range(self.n_episodes_per_iter):
                new_desired_reward, new_desired_horizon = self.sampling_exploration()
                generated_episode = self.generate_episode(new_desired_reward, new_desired_horizon)
                self.buffer.add_sample(generated_episode[0], generated_episode[1], generated_episode[2])
            new_desired_reward, new_desired_horizon = self.sampling_exploration()
            desired_rewards_history.append(new_desired_reward.item())
            horizon_history.append(new_desired_horizon.item())
            ep_rewards = self.evaluate(new_desired_reward, new_desired_horizon)
            all_rewards.append(ep_rewards)
            average_100_reward.append(np.mean(all_rewards[-100:]))
            print(
                f"\rEpisode: {ep} | Rewards: {ep_rewards:.2f} | Mean_100_Rewards: {np.mean(all_rewards[-100:]):.2f} | Loss: {np.mean(loss_buffer):.2f}",
                end="", flush=True)
            if ep % 100 == 0:
                print(
                    f"\rEpisode: {ep} | Rewards: {ep_rewards:.2f} | Mean_100_Rewards: {np.mean(all_rewards[-100:]):.2f} | Loss: {np.mean(loss_buffer):.2f}")
        return all_rewards, average_100_reward, desired_rewards_history, horizon_history, losses

    def train_and_plot(self, max_episodes=10):
        all_rewards, average_100_reward, desired_rewards_history, horizon_history, losses = self.run_upside_down(
            max_episodes)
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        plt.title("Rewards")
        plt.plot(all_rewards, label="rewards")
        plt.plot(average_100_reward, label="average100")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.title("Loss")
        plt.plot(losses)
        plt.subplot(2, 2, 3)
        plt.title("Desired Rewards")
        plt.plot(desired_rewards_history)
        plt.subplot(2, 2, 4)
        plt.title("Desired Horizon")
        plt.plot(horizon_history)
        plt.show()
