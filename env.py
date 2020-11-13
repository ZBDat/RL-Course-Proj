# the scenario of the exercises

import numpy as np


class GridWorld:
    """
    """

    def __init__(self, reward_map: np.ndarray, direction_movement_probs: np.ndarray, gamma=0.9):
        self.gamma = gamma
        self.reward_map = reward_map
        self.directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.direction_movement_probs = direction_movement_probs
