import random
import time
import math
import numpy as np
from itertools import count
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import matplotlib.pyplot as plt

from Environment import CartPoleEnvironment

# hyperparameters
EPISODES = 1000
EPS_START = 1.0
EPS_END = 0.01
gamma = 0.99
batch_size = 256
lr = 0.001
update_target = 100
log_interval = 10
replay_memory_capacity = 10000
TAU = 0.001
sigma_zero = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def append(self, state, next_state, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, mask))
        self.memory[self.position] = Transition(state, next_state, action, reward, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

class DQNAgent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQNAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # Duel Net
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_adv = nn.Linear(32, num_outputs)
        self.fc_val = nn.Linear(32, 1)
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        adv = self.fc_adv(x)
        adv = adv.view(-1, self.num_outputs)
        val = self.fc_val(x)
        val = val.view(-1, 1)

        q_value = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_value

    @classmethod
    def train_model(cls, policy_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = policy_net(states).squeeze(1)
        _, action_policy_net = policy_net(next_states).squeeze(1).max(1)
        next_pred = target_net(next_states).squeeze(1)
        state_action_values = torch.sum(pred.mul(actions), dim=1)

        expected_state_action_values = rewards + masks * gamma * next_pred.gather(1, action_policy_net.unsqueeze(1)).squeeze(1)

        # backpropagation of loss to NN
        loss = F.mse_loss(state_action_values, expected_state_action_values.detach())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss
    
    def get_action(self, input):
        q_value = self.forward(input)
        _, action = torch.max(q_value, 1)
        return action.numpy()[0]
    
    def reset_noise(self):
        self.fc_adv.reset_noise()

def choose_action(state, policy_net, epsilon):
    with torch.no_grad():
        greedy_action = policy_net.get_action(state)
        greedy_action -= 10
    random_action = np.random.randint(-10, 10)
    return random_action if np.random.rand() <= epsilon else greedy_action

def update_target_model(policy_net, target_net):
    """ Update target network with the model weights """
    # Extract parameters  
    model_params = policy_net.named_parameters()
    target_params = target_net.named_parameters()
    
    updated_params = dict(target_params)

    for model_name, model_param in model_params:
        if model_name in target_params:
            # Update parameter
            updated_params[model_name].data.copy_((TAU)*model_param.data + (1-TAU)*target_params[model_param].data)

    #target_net.load_state_dict(policy_net.state_dict())
    target_net.load_state_dict(updated_params)

def train():
    final = []
    env = CartPoleEnvironment()
    np.random.seed(450)
    torch.manual_seed(450)

    num_inputs = 4
    num_actions = 21

    policy_net = DQNAgent(num_inputs, num_actions).to(device)
    target_net = DQNAgent(num_inputs, num_actions).to(device)
    update_target_model(policy_net, target_net)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    policy_net.to(device)
    target_net.to(device)
    policy_net.train()
    target_net.train()
    memory = ReplayMemory(replay_memory_capacity)

    epsilon = 1.0
    steps = 0
    loss = 0

    start_time = time.time()

    for e in range(EPISODES):
        episode_reward = 0
        episode_steps = 0
        episode_loss = 0
        done = False
        state = env.reset()
        env.clear()
        eps_start_time = time.perf_counter()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        for t in range(500):
            episode_steps += 1
            steps += 1
            action = choose_action(state, policy_net, epsilon)
            next_state, reward, done = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done else -1
            if action < -10 or action > 10:
                action = np.clip(action, -10, 10)
            action_idx = np.zeros(21)
            action_idx[action + 10] = 1
            # store transitions in replay memory
            memory.append(state, next_state, action_idx, reward, mask)
            state = next_state

            # initial exploration
            if steps > 1000:
                epsilon -= 0.00005
                epsilon = max(epsilon, EPS_END)
                # Sample experiences from the agent's memory
                batch = memory.sample(batch_size)
                loss = DQNAgent.train_model(policy_net, target_net, optimizer, batch)
            
            episode_reward += reward
            episode_loss += loss

            if done:
                break
        
        # update epsilon after each episode
        # epsilon = (EPS_END - EPS_START) * (e / EPISODES) + EPS_START
        
        #  update the target network
        if steps % update_target == 0:
            # Update target network weights using replay memory
            update_target_model(policy_net, target_net)

        if e % log_interval == 0:
            print("episode: {}/{} | total reward: {}, average_reward:{}, e: {:.2} | time: {}".format(e, EPISODES, episode_reward, episode_reward / episode_steps, epsilon, time.perf_counter() - eps_start_time))

        final.append(episode_reward)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Complete')
    plot_res(final, title='DQN')
    env.render()

def random_search():
    final = []
    env = CartPoleEnvironment()

    for e in range(EPISODES):
        state = env.reset()
        env.clear()
        done = False
        episode_reward = 0
        while not done:
            action = np.random.randint(-10, 10) * np.random.choice([-1,1])
            next_state, reward, done = env.step(action)
            episode_reward += reward
        if e % log_interval == 0:
            print("episode: {}/{} | score: {} ".format(e, EPISODES, episode_reward))
        final.append(episode_reward)
    plot_res(final, "Random Search")
    env.render()

def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='total reward per episode')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total Reward')
    x = range(len(values))
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    ax[0].legend()
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].set_xlabel('Total Reward Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    train()
    #random_search()
    
