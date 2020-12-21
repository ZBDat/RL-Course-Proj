import random
import time
from typing import Tuple, List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi


class CartPoleEnvironment:

    def __init__(self, cart_mass=0.5, pole_mass=0.5, pole_length=0.6, friction=0.1, action_range=(-10, 10),
                 action_interval=0.1, update_interval=0.01):
        """
        Physical parameters for the simulation
        :param cart_mass: mass of the translational moving part
        :param pole_mass: mass of the pole, concentrated at the end
        :param pole_length: length of the pole
        :param friction: frictional coefficient b
        :param action_range: range of the external force F
        :param action_interval: the time interval for the new action selection
        :param update_interval: the time interval for the state update
        """
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.friction = friction
        self.gravityConstant = 9.8

        self._displacementLimit = (-6, 6)
        self._velocityLimit = (-10, 10)
        self._angleLimit = (-pi, pi)
        self._angularVelocityLimit = (-10, 10)

        self._action_range = action_range
        self._action_interval = action_interval
        self._update_interval = update_interval

        self.state: Tuple[float, float, float, float] or None = None
        self.time: float or None = None
        self.rewards: List[float] or None = None

    def reset(self) -> Tuple[float, float, float, float]:
        """
        The function to reset the state of the simulator
        :return: self.state
        """
        self.state = (0, 0, pi, 0)
        self.time = 0
        self.rewards = []

        return self.state

    def step(self, action) -> Tuple[Tuple[float, float, float, float], float]:
        """
        The function to calculate the state with a given action
        :return: self.state, reward
        """
        assert self.state is not None, "state is not defined!"

        if action < min(self._action_range) or action > max(self._action_range):
            action = np.clip(action, min(self._action_range), max(self._action_range))
            print("action out of range, clipped")

        x, v, theta, omega = self.state
        m1 = self.cart_mass
        m2 = self.pole_mass
        l = self.pole_length
        b = self.friction
        g = self.gravityConstant

        # linear acceleration
        alpha = (2 * m2 * l * omega ** 2 * np.sin(theta) - 3 * m2 * g * np.sin(theta) * np.cos(
            theta) + 4 * action - 4 * b * v) / (4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2)

        # angular acceleration
        beta =


def run_test(environment: CartPoleEnvironment):
    """
    Use this function to test if the environment setup is functioning
    :param environment:
    :return:
    """
    ...


if __name__ == '__main__':
    environment = CartPoleEnvironment(cart_mass=0.5, pole_mass=0.5, pole_length=0.6, friction=0.1,
                                      action_range=(-10, 10),
                                      action_interval=0.1, update_interval=0.01)
    run_test(environment)
