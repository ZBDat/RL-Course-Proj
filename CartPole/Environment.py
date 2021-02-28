from typing import Tuple, List, Dict, Any

import numpy as np
from numpy import pi

from Renderer import Renderer


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
        self.x_threshold = 6

        self._action_range = action_range
        self._action_interval = action_interval
        self._update_interval = update_interval

        self.state: Tuple[float, float, float, float] or None = None
        self.time: float or None = None
        self.rewards: List[float] or None = []
        self.state_list: List[Tuple[float, float, float, float]] or None = []

        self.state_reward_dict = {}

        self.use_renderer = use_renderer
        self.renderer = None

    def reset(self) -> Tuple[float, float, float, float]:
        """
        The function to reset the state of the simulator
        :return: self.state
        """
        self.state = tuple(np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02])))
        self.time = 0
        self.rewards.append(self.get_reward(self.state))
        self.state_list.append(self.state)

        self.state_reward_dict[self.state_list[0]] = self.rewards[0]

        return self.state

    def clear(self):
        """
        clear the stored state list
        :return:
        """
        self.rewards = []
        self.state_list = []
        self.state_reward_dict = {}

    def get_reward(self, state: Tuple[float, float, float, float]):
        """
        Calculate the reward corresponding to a state
        :param state: state of the environment
        :return: reward
        """
        x, v, theta, omega = state

        j = np.array([x, np.sin(theta), np.cos(theta)])
        j_target = np.array([0, 0, 1])
        quad = lambda A, vec: (np.dot(vec.T, np.dot(A, vec)))  # quadratic form
        reward = -1 * (1 - np.exp(-0.5 * quad(self.rewardMatrix, (j - j_target))))

        return reward

    def step(self, action) -> Tuple[Tuple[float, float, float, float], float, bool]:
        """
        The function to calculate the state with a given action
        :return: self.state, reward
        """
        assert self.state is not None, "state is not defined!"

        # Valid action
        if action < min(self._action_range) or action > max(self._action_range):
            action = np.clip(action, min(self._action_range), max(self._action_range))
            print("action out of range, clipped")

        x, v, theta, omega = self.state

        v_min, v_max = self._velocityLimit
        x_min, x_max = self._displacementLimit
        omega_min, omega_max = self._angularVelocityLimit
        theta_min, theta_max = self._angleLimit

        m1 = self.cart_mass
        m2 = self.pole_mass
        l = self.pole_length
        b = self.friction
        g = self.gravityConstant

        # calculate the reward
        reward = self.get_reward(self.state)

        remaining_time = self._action_interval
        delta_t = self._update_interval

        while remaining_time > 0:
            if delta_t < remaining_time:
                delta_t = remaining_time

            # linear acceleration
            alpha = (2 * m2 * l * omega ** 2 * np.sin(theta) - 3 * m2 * g * np.sin(theta) * np.cos(
                theta) + 4 * action - 4 * b * v) / (4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2)

            # angular acceleration
            beta = (3 * m2 * l * omega ** 2 * np.sin(theta) * np.cos(theta) - 6 * (m1 + m2) * g * np.sin(theta) + 6 * (
                    action - b * v) * np.cos(
                theta)) / (4 * l * (m1 + m2) - 3 * m2 * l * np.cos(theta) ** 2)

            # Euler method
            # velocity
            v = v + delta_t * alpha
            if v > v_max or v < v_min:
                v = np.clip(v, v_min, v_max)

            # displacement
            x = x + delta_t * v + 1 / 2 * delta_t ** 2 * alpha

            # angular velocity
            omega = omega + delta_t * beta
            if omega > omega_max or omega < omega_min:
                omega = np.clip(omega, omega_min, omega_max)

            # angular displacement
            theta = theta + delta_t * omega + 1 / 2 * delta_t ** 2 * beta
            theta = theta % (2 * pi)
            if theta > theta_max:
                theta -= 2 * theta_max

            remaining_time -= delta_t

        # give the next state
        next_state = (x, v, theta, omega)
        self.state = next_state

        done = self._terminal(x)

        # reward_theta = (np.cos(theta)+1.0)/2.0
        # Reward_x is 0 when cart is at the edge of the screen, 1 when it's in the centre
        # reward_x = np.cos((x / 3) * (np.pi / 2.0))
        # reward *= reward_x

        self.state_list.append(self.state)
        self.rewards.append(reward)
        self.state_reward_dict[self.state] = reward

        return next_state, reward, done

    # episode termination
    def _terminal(self, x):
        return(bool(abs(x) > self.x_threshold))

    def render(self):
        if self.use_renderer:
            self.renderer = Renderer(length=self.pole_length, x_range=self._displacementLimit)
        self.renderer.animate(state_list=self.state_list, reward_list=self.rewards)


def run_test():
    """
    Use this function to test if the environment setup is functioning
    :return: None
    """
    environment = CartPoleEnvironment(cart_mass=0.5, pole_mass=0.5, pole_length=0.6, friction=0.1,
                                      action_range=(-10, 10),
                                      action_interval=0.1, update_interval=0.01)

    actions = np.random.randint(-2, 2, 200)
    environment.reset()

    for action in actions:
        environment.step(action)

    environment.render()


if __name__ == '__main__':
    np.random.seed(0)
    run_test()
