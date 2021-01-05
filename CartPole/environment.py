import random

from typing import Tuple, List, Dict, Any

import numpy as np
from numpy import pi

from CartPole.Renderer import Renderer


class CartPoleEnvironment:

    def __init__(self, cart_mass=0.5, pole_mass=0.5, pole_length=0.6, friction=0.1, action_range=(-10, 10),
                 action_interval=0.1, update_interval=0.01, use_renderer=True):
        """
        Physical parameters for the simulation
        :param cart_mass: mass of the translational moving part
        :param pole_mass: mass of the pole, concentrated at the end
        :param pole_length: length of the pole
        :param friction: frictional coefficient b
        :param action_range: range of the external force F
        :param action_interval: the time interval for the new action selection
        :param update_interval: the time interval for the state update
        :param use_renderer: whether to use the renderer to animate the results
        """
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.friction = friction
        self.gravityConstant = 9.8
        self.rewardMatrix = np.array([[1, self.pole_length, 0], [self.pole_length, self.pole_length ** 2, 0],
                                      [0, 0, self.pole_length ** 2]])

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

        if use_renderer:
            self.renderer = Renderer(length=self.pole_length, radius=0.1, dt=0.1)

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

        v_min, v_max = (-10, 10)
        x_min, x_max = (-6, 6)
        omega_min, omega_max = (-10, 10)
        theta_min, theta_max = (-pi, pi)

        m1 = self.cart_mass
        m2 = self.pole_mass
        l = self.pole_length
        b = self.friction
        g = self.gravityConstant
        delta_t = self._update_interval

        # calculate the reward
        j = np.array([x, np.sin(theta), np.cos(theta)])
        j_target = np.array([0, 0, 1])
        quad = lambda A, vec: (np.dot(vec.T, np.dot(A, vec)))  # quadratic form
        reward = -1 * (1 - np.exp(-0.5 * quad(self.rewardMatrix, (j - j_target))))

        # linear acceleration
        alpha = (2 * m2 * l * omega ** 2 * np.sin(theta) - 3 * m2 * g * np.sin(theta) * np.cos(
            theta) + 4 * action - 4 * b * v) / (4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2)

        # angular acceleration
        beta = (3 * m2 * l * omega ** 2 * np.sin(theta) * np.cos(theta) - 6 * (m1 + m2) * g * np.sin(theta) + 6 * (
                action - b * v) * np.cos(
            theta)) / (4 * l * (m1 + m2) - 3 * m2 * l * np.cos(theta) ** 2)

        # Euler method
        v = v + delta_t * alpha
        if v > v_max or v < v_min:
            v = np.clip(v, v_min, v_max)

        x = x + delta_t * v + 1 / 2 * delta_t ** 2 * alpha
        if x > x_max or x < x_min:
            x = np.clip(x, x_min, x_max)

        omega = omega + delta_t * beta
        if omega > omega_max or omega < omega_min:
            omega = np.clip(omega, omega_min, omega_max)

        theta = theta + delta_t * omega + 1 / 2 * delta_t ** 2 * beta
        if theta > theta_max or theta < theta_min:
            theta = np.clip(theta, theta_min, theta_max)

        # give the next state and reward
        next_state = x, v, theta, omega
        self.state = next_state
        self.rewards.append(reward)

        return next_state, reward

    def render(self):
        ...


def run_test():
    """
    Use this function to test if the environment setup is functioning
    :param environment:
    :return:
    """
    environment = CartPoleEnvironment(cart_mass=0.5, pole_mass=0.5, pole_length=0.6, friction=0.1,
                                      action_range=(-10, 10),
                                      action_interval=0.1, update_interval=0.01)

    environment.reset()
    environment.step(5)


if __name__ == '__main__':
    run_test()
