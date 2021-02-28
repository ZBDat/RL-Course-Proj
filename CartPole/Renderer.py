from typing import Tuple, List

from matplotlib import pyplot as plt
from matplotlib import animation

import numpy as np


class Renderer:

    def __init__(self, length=0.6, radius=0.1, max_time=None, x_range=(-6.5, 6.5), y_range=(-1, 1)):
        self.length = length
        self.radius = radius
        self.max_time = max_time

        self.fig = plt.figure()
        self.fig.set_dpi(100)

        self.simulation_ax = self.fig.add_subplot(211, xlim=x_range, ylim=y_range)
        self.simulation_ax.set_title('Cart-Pole Simulation')

        self.reward_ax = self.fig.add_subplot(212, ylim=(-1.5, 0.5))
        self.reward_ax.set_ylabel('Reward')
        self.reward_ax.set_xlabel('Steps')

        self.simulation_ax.grid()
        self.reward_ax.grid()

        self.patch1 = plt.Circle((0, 0), 0.1, fc='black', ec='black')
        self.patch2 = plt.Line2D((0, self.length), (0, 0), lw=1.5)
        self.patch3 = plt.Rectangle((-0.25, -0.125), 0.5, 0.25, fc='white', ec='black')
        self.text = self.simulation_ax.text(0.8, 0.8, '', transform=self.simulation_ax.transAxes)

        self.reward_curve, = plt.plot([0, ], [0, ], lw=2)

    def initialize_plot(self):
        """
        The starting frame of the animation, required by the FuncAnimation method
        :return:
        """
        self.patch1.center = (0, -0.6)
        self.patch2.set_data(np.array([0, 0]), np.array([0, 0.6]))
        self.patch3.set_x(-0.5)
        self.text.set_text('')
        self.reward_curve.set_data([0, ], [0, ])

        self.simulation_ax.add_artist(self.patch1)
        self.simulation_ax.add_artist(self.patch2)
        self.simulation_ax.add_artist(self.patch3)

        return self.patch1, self.patch2, self.patch3, self.text,

    def plot_shapes(self, state):
        """
        Repeated frames of the animation, required by the FuncAnimation method
        :param state: state of the cart-pole
        :return: any plotted objects
        """
        x2, y2 = self.patch1.center
        x1, v, theta, omega = state
        y1 = 0
        x2 = x1 + self.length * np.sin(theta)
        y2 = y1 + self.length * np.cos(theta)
        self.patch1.center = (x2, y2)
        self.patch2.set_data(np.array([x1, x2]), np.array([y1, y2]))
        self.patch3.set_x(x1 - 0.25)
        self.text.set_text('x = %.1f\ntheta = %.1f' % (x1, theta))

        return self.patch1, self.patch2, self.patch3, self.text,

    def plot_rewards(self, reward):
        self.reward_curve.set_xdata(np.append(self.reward_curve.get_xdata(), self.reward_curve.get_xdata()[-1] + 1))
        self.reward_curve.set_ydata(np.append(self.reward_curve.get_ydata(), reward))

        return self.reward_curve,

    def animate(self, state_list: List = None, reward_list: List = None):
        """
        Animation. the list of state and reward should have the same length
        :param reward_list: list recording the rewards
        :param state_list: list recording the states
        :return: a FuncAnimation object
        """

        # Tweaking option: set interval higher: slow down the animation
        ani1 = animation.FuncAnimation(self.fig, self.plot_shapes, frames=state_list,
                                       interval=30, blit=True, init_func=self.initialize_plot)

        if reward_list:
            self.reward_ax.set_xlim(0, len(reward_list))
            ani2 = animation.FuncAnimation(self.fig, self.plot_rewards, frames=reward_list,
                                       interval=50, blit=True, init_func=self.initialize_plot)

        else:
            ani2 = None

        plt.show()

        return ani1, ani2

    def plot_accumulated_reward(self, accumulated_reward: List):
        fig2 = plt.figure()
        plt.plot(accumulated_reward)


if __name__ == "__main__":
    render = Renderer()
    x = np.linspace(-1, 1, 50)
    v = 0
    theta = np.linspace(np.pi, 0, 50)
    omega = 0
    state_list = []
    for i in range(len(x)):
        state = (x[i], v, theta[i], omega)
        state_list.append(state)

    reward_list = [0] * len(state_list)

    render.animate(state_list, reward_list)
