from typing import Tuple

import numpy as np
import random


class GridEnvironment():
    # TODO For Mun Seng
    # Implement Env

    def __init__(self, grid_size=(3, 4)):
        """
        :param state: Initial state
        :param shape: grid size
        """
        # available actions
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        self.actionsList = list(self.actions.keys())
        self.walls = [(1, 1)]
        # Total number of states
        self.shape = grid_size
        self.nS = grid_size[0] * grid_size[1]
        self.nA = len(self.actions)
        
    def get_reward(self, state):
        # reward function
        if state == (0, 3):
            reward = 1
        elif state == (1, 3):
            reward = -100
        else:
            reward = 0
        return reward

    def step(self, state, action):
        """"
        Move the agent in the specified direction. If the agent is at a border or 
        hit an obstacle, it stays still at the current state.
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        :param action: up, down, left, right
        :return: Tuple of next state and reward
        """

        action = self.actions.get(action)
        # move to next state base on action
        next_state = (state[0] + action[0], state[1] + action[1])

        # check if next state legal
        # stay on current state if bump into obstacle
        if next_state in self.walls:
            next_state = state
        # out of bounds
        elif next_state[0] < 0 or next_state[0] >= self.shape[0]:
            next_state = state
        elif next_state[1] < 0 or next_state[1] >= self.shape[1]:
            next_state = state

        reward = self.get_reward(next_state)

        return next_state, reward
    
    def turn_right(self, action):
        return self.actionsList[(self.actionsList.index(action) + 2) % len(self.actionsList)]
    
    def turn_left(self, action):
        return self.actionsList[(self.actionsList.index(action) + 3) % len(self.actionsList)]

    def transition_matrix(self, state, action):
        """
        T: [action_prob, (next_state, reward)]
        """
        t1 = [0.8, self.step(state, action)]
        t2 = [0.1, self.step(state, self.turn_right(action))]
        t3 = [0.1, self.step(state, self.turn_left(action))]

        return [t1, t2, t3]
    
    def bellman_expectation(self, state_values, state, action, discount=0.9):
        """
        Makes a one step lookahead and applies the bellman expectation equation to the state
        :return list of calculated state values wrt the input action [(action, state_values)]
        """
        calculated_values = list()
        value = 0
        trans_matrix = self.transition_matrix(state, action)

        for prob, matrix in trans_matrix:
            reward = matrix[1]
            next_state = matrix[0]

            # sum of action defined by policy
            value += prob * (reward + discount * state_values[next_state])

        # append sums to calculate max state value
        calculated_values.append((action, value))

        return calculated_values
    
    def get_all_states(self):
        """
        :return: grid = Grid Object, l = list of accessible states
        """
        initial_grid = np.zeros(shape=self.shape)
        l = [(i, j) for i, y in enumerate(initial_grid) for j, x in enumerate(y) if (i, j) not in self.walls]

        return l
    
    def print_policy(self, p):
        for col in range(self.shape[0]):
            for row in range(self.shape[1]):
                a = p.get((col, row), ' ')
                if a == ' ':
                    print(" # |", end='')
                else:
                    print(" {a} |".format(a=a+1), end='')
            print("")


def policy_iteration():
    """
    :param policy: dict where the key identify a state (i, j) and the item identifies the actions.
                    Example output:
                    {(0, 0): 3, (0, 1): 3, (0, 2): 3, (0, 3): 0, (1, 0): 0, (1, 2): 2, (1, 3): 2, (2, 0): 0, (2, 1): 2, (2, 2): 2, (2, 3): 3}
    :param value: dict where the key identify a state (i, j) and the item identifies the state values.

    """
    discount = 0.9
    threshold = 0.1
    env = GridEnvironment()

    list_of_states = env.get_all_states()

    policy = {s: random.choice(env.actionsList) for s in list_of_states}

    iteration = 0

    # State values
    V = {s: 0.0 for s in list_of_states}
    print(V)

    while True:
        iteration += 1
        # 1 Policy evaluation
        delta = 0
        for s in list_of_states:
            old_v = V[s]
            a = policy[s]
            action_value = [v for _, v in env.bellman_expectation(V, s, a, discount)]
            V[s] = action_value[0]
            delta = np.abs(old_v - V[s]).max()

        # Stopping criteria
        if delta < threshold: break

        # Policy Improvement
        for s in list_of_states:
            if s in policy:
                new_a = None
                best_value = float('-inf')
                for a in env.actionsList:
                    action_values = env.bellman_expectation(V, s, a, discount)

                    for v in action_values:
                        if v[1] > best_value:
                            best_value = v[1]
                            new_a = v[0]

                    new_a_1 = np.argmax(action_values)
                
                if new_a != policy[s]: policy[s] = new_a
        

    print("============= FINAL RESULT ============")
    print("Policy Iteration")
    print("Iterations: " + str(iteration))
    print("Optimal Policy:")
    env.print_policy(policy)


def exercise_1():
    # TODO For Peter
    # Implement Q-Learning
    pass


def exercise_2():
    # TODO For Zhaobo
    # Implement SARSA
    pass


if __name__ == "__main__":
    policy_iteration()
