from typing import Tuple

import numpy as np


class GridEnvironment():
    # TODO For Mun Seng
    # Implement Env

    def __init__(self, state: np.ndarray):
        """
        :param state: Initial state
        """
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float]:
        """
        :param action:
        :return: Tuple of next state and reward
        """
        pass


def exercise_1():
    # TODO For Peter
    # Implement Q-Learning
    pass


def exercise_2():
    # TODO For Zhaobo
    # Implement SARSA
    pass


if __name__ == "__main__":
    exercise_1()
    exercise_2()
