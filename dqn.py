# adopted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from timing import Timing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, stride=2)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, stride=2)))
        linear_input_size = convw * convh * 16

        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.reshape(x.size(0), -1))

class RandomActor:

    def __init__(self, screen_height, screen_width, n_channels, n_actions, buffer_size=5_000):
        self.n_actions = n_actions
        self.input_shape = (screen_height, screen_width, n_channels)

        self.memory = ReplayMemory(buffer_size)
        self.last_observation = self.format_observation(np.zeros(self.input_shape))

    def format_observation(self, screen):
        return torch.from_numpy(screen.transpose((2, 0, 1))).unsqueeze(0).to(device, dtype=torch.float)

    def new_episode(self, observation):
        self.last_observation = self.format_observation(observation)

    def step(self, env, timing):

        # Select and perform an action
        with Timing(timing, 'time_choose_act'):
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        with Timing(timing, 'time_perform_act'):
            raw_observation, reward, done, info = env.step(action.item())

        observation = self.format_observation(raw_observation)
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = observation
        else:
            next_state = None

        # Store the transition in memory
        self.memory.push(self.last_observation, action, next_state, reward)
        self.last_observation = observation

        return raw_observation, reward, done, info, action


class DQNActor:

    def __init__(self, screen_height, screen_width, n_channels, n_actions, eps_start=1,
                 eps_end=0.01, eps_test=0.001, eps_decay=250_000, eps_const=None, gamma=0.999, 
                 batch_size=32, target_update=2000, learning_rate=0.00025, buffer_size=1_000_000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_const = eps_const
        if eps_const is None:
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay = eps_decay
            self.eps_threshold = eps_start
        else:
            self.eps_threshold = eps_const
        self.batch_size = batch_size
        self.target_update = target_update

        self.input_shape = (screen_height, screen_width, n_channels)
        self.policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(buffer_size)

        self.steps_done = 0

        self.last_observation = self.format_observation(np.zeros(self.input_shape))

    # Reshape to torch order CHW and add a batch dimension (BCHW)
    def format_observation(self, screen):
        return torch.from_numpy(screen.transpose((2, 0, 1))).unsqueeze(0).to(device, dtype=torch.float)

    def select_action(self, observation):
        sample = random.random()
        if self.eps_const:
            eps_threshold = self.eps_const
        else:
            eps_threshold = self.eps_start - self.steps_done * (self.eps_start - self.eps_end) / self.eps_decay
            eps_threshold = max(self.eps_end, eps_threshold) # make sure we don't go lower than eps_endd
        self.eps_threshold = eps_threshold
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.policy_net.eval()
                result = self.policy_net(observation).max(1)[1].view(1, 1)
                self.policy_net.train()
                return result
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self, i_episode):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            wandb.log({
                'target_update': i_episode,
            }, commit=False)
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def new_episode(self, observation):
        self.last_observation = self.format_observation(observation)

    def step(self, env, i_episode, timing):

        # Select and perform an action
        with Timing(timing, 'time_choose_act'):
            action = self.select_action(self.last_observation)

        with Timing(timing, 'time_perform_act'):
            raw_observation, reward, done, info = env.step(action.item())

        observation = self.format_observation(raw_observation)
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = observation
        else:
            next_state = None

        self.memory.push(self.last_observation, action, next_state, reward)
        self.last_observation = observation


        # Perform one step of the optimization (on the target network)
        with Timing(timing, 'time_optimize_model'):
            self.optimize_model(i_episode)

        return raw_observation, reward, done, info, action

