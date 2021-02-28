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
EPISODES = 2000

# epsilon decay schedule
eps_start = 1.0
eps_min = 0.1
eps_decay = 0.99995

gamma = 0.99 # discount factor
batch_size = 256 # minibatch size
lr = 0.001 # learning rate
update_target = 100  # how frequently update target network
log_interval = 10
replay_memory_size = 10000 # replay buffer size (maximum number of experiences stored in replay memory)
TAU = 0.001

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

        state_values = policy_net(states).squeeze(1)
        _, action_policy_net = policy_net(next_states).squeeze(1).max(1)
        next_state_values = target_net(next_states).squeeze(1)
        state_action_values = torch.sum(state_values.mul(actions), dim=1)

        expected_state_action_values = rewards + masks * gamma * next_state_values.gather(1, action_policy_net.unsqueeze(1)).squeeze(1)

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

def choose_action(state, policy_net, epsilon):
    with torch.no_grad():
        greedy_action = policy_net.get_action(state)
        greedy_action = (greedy_action * 2) -10
    # random_action = np.random.randint(-10, 10)
    random_action = random.randrange(-10, 10, 2)
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
    env = CartPoleEnvironment()
    np.random.seed(542)
    torch.manual_seed(542)

    num_inputs = 4
    num_actions = 11

    policy_net = DQNAgent(num_inputs, num_actions).to(device)
    target_net = DQNAgent(num_inputs, num_actions).to(device)
    update_target_model(policy_net, target_net)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    policy_net.to(device)
    target_net.to(device)
    policy_net.train()
    target_net.train()
    memory = ReplayMemory(replay_memory_size)

    epsilon = eps_start
    steps = 0
    loss = 0

    start_time = time.time()
    total_rewards = []
    average_rewards = []
    best_mean_reward = None
    mean_reward_bound = 0

    for e in range(EPISODES):
        episode_steps = 0
        episode_loss = 0
        done = False
        state = env.reset()
        env.clear()
        eps_start_time = time.perf_counter()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        average_reward = 0
        total_reward = 0

        for t in range(1000):
            episode_steps += 1
            steps += 1
            action = choose_action(state, policy_net, epsilon)
            next_state, reward, done = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done else -100

            if reward is not None and steps > 1000:
                total_reward += reward
                total_rewards.append(total_reward)

                average_reward += (total_reward / episode_steps)
                average_rewards.append(average_reward)
                
                mean_reward = np.mean(total_rewards[-100:])

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    best_mean_reward = mean_reward
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f" % (best_mean_reward))
                
                # early stopping
                if mean_reward > mean_reward_bound:
                    print("Complete! Solved in %d episodes!" % episode_steps)
                    break

            action_idx = np.zeros(11)
            action_ind = int((action + 10) / 2 )
            action_idx[action_ind] = 1
            # store transitions in replay memory
            memory.append(state, next_state, action_idx, reward, mask)
            state = next_state

            # initial exploration
            if steps > 10000:
                # epsilon -= 0.00005
                epsilon = max(epsilon*eps_decay, eps_min)
                # Sample experiences from the agent's memory
                batch = memory.sample(batch_size)
                loss = DQNAgent.train_model(policy_net, target_net, optimizer, batch)
            
            episode_loss += loss

            if done:
                break
        
        # update epsilon after each episode
        # epsilon = max(epsilon*eps_decay, eps_min)
        
        #  update the target network
        if steps % update_target == 0:
            # Update target network weights using replay memory
            update_target_model(policy_net, target_net)

        if e % log_interval == 0:
            print("episode: {}/{} | total reward: {}, average_reward:{}, e: {:.2} | eps_training_time: {}". \
                format(e, EPISODES, total_reward, total_reward / episode_steps, epsilon, time.perf_counter() - eps_start_time))
    
    print("--- Training time: %s seconds ---" % (time.time() - start_time))
    plot_res(total_rewards, title='DDQN with soft update')
    env.render()

def plot_res(values, values2, title=''):   
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
        ax[0].plot(x,p(x),"r--", label='trend')
    except:
        print('')
    ax[0].legend()
    
    # Plot the histogram of results
    ax[1].plot(values2)
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Average Reward per Episode')
    plt.show()

if __name__ == "__main__":
    train()
    
