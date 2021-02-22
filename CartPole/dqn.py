import random
import math
import numpy as np
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from drawnow import drawnow
import matplotlib.pyplot as plt

from Environment import CartPoleEnvironment

last_score_plot = [0]
avg_score_plot = [0]

def draw_fig():
    plt.title('reward')
    plt.plot(last_score_plot, '-')
    plt.plot(avg_score_plot, 'r-')

EPISODES = 500
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
# hyperparameters
gamma = 0.90
batch_size = 512
lr = 0.001
log_interval = 10
update_target = 10
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def append(self, item):
        self.memory.append(item)

    def sample(self, batch_size):
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]

    def __len__(self):
        return len(self.memory)

class DQNAgent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQNAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        init.xavier_normal_(self.fc1.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

def train_model(states, actions, next_states, rewards, dones, policy_net, target_net, optimizer):
    state_action_values = policy_net(states).gather(1, actions.unsqueeze(1).long()).squeeze()
    next_state_values = torch.max(target_net(next_states), dim=1)[0].detach()
    expected_state_action_values = rewards + next_state_values * (1 - dones) * gamma

    # backpropagation of loss to NN
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def choose_action(state, policy_net, epsilon):
    with torch.no_grad():
        greedy_action = torch.argmax(policy_net(state), dim=1).item()
        greedy_action -= 20
        random_action = np.random.randint(-10, 10)
    return random_action if np.random.rand() < epsilon else greedy_action

def update_target_model(policy_net, target_net):
    # Target <- Net
    target_net.load_state_dict(policy_net.state_dict())

def cartpole():
    env = CartPoleEnvironment()
    np.random.seed(400)
    torch.manual_seed(400)

    num_inputs = 4
    num_actions = 21

    policy_net = DQNAgent(num_inputs, num_actions).to(device)
    target_net = DQNAgent(num_inputs, num_actions).to(device)
    update_target_model(policy_net, target_net)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-4)

    #policy_net.train()
    #target_net.train()
    memory = ReplayMemory(replay_memory_capacity)

    for e in range(EPISODES):
        episode_durations = 0
        done = False
        state = env.reset()
        epsilon = (EPS_END- EPS_START) * (e / EPISODES) + EPS_START

        for t in count():
            action = choose_action(torch.tensor(state).float()[None, :], policy_net, epsilon)
            next_state, reward, done = env.step(action)

            if action < -10 or action > 10:
                action = np.clip(action, -10, 10)
            action += 10
            memory.append([state, action, next_state, reward, done])
            state = next_state

            if len(memory) > batch_size:
                states, actions, next_states, rewards, dones = \
                    map(lambda x: torch.tensor(x).float(), zip(*memory.sample(batch_size)))

                train_model(states, actions, next_states, rewards, dones, policy_net, target_net, optimizer)
            
            if done:
                episode_durations = t + 1
                avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_durations * 0.01)
                last_score_plot.append(episode_durations)
                drawnow(draw_fig)
                break

        if e % update_target == 0:
            update_target_model(policy_net, target_net)

        if e % log_interval == 0:
            print("episode: {}/{} | avg_score: {}, e: {:.2}".format(e, EPISODES, reward, epsilon))
    
    print('Complete')
    env.render()

if __name__ == "__main__":
    cartpole()
