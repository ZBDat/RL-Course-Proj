from typing import Tuple, List

from matplotlib import pyplot as plt
from matplotlib import animation

import numpy as np


class Renderer:

    def __init__(self, length=0.6, radius=0.1, dt=0.1, simulation_ax=None, reward_ax=None):
        """
        The class to visualize the cart-pole problem
        :param length: length of the pole
        :param radius: radius of the ball on the pole
        :param dt: time step for update
        :param simulation_ax: subplot for the animation
        :param reward_ax: subplot for the reward curve
        """
        self.fig = plt.figure()
        self._length = length
        self._radius = radius
        self._dt = dt

        self._circle: plt.Circle or None = None
        self._cart: plt.Rectangle or None = None
        self._text: plt.Text or None = None

        self._simulation_ax: plt.Axes = simulation_ax
        self._reward_ax: plt.Axes = reward_ax

        self._simulation_ax = self.fig.add_subplot(211, autoscale_on=True, aspect='equal', xlim=(-2, 2), ylim=(-1, 1))
        self._reward_ax = self.fig.add_subplot(212, autoscale_on=True, aspect='equal', xlim=(-3, 3), ylim=(-3, 3))

        self.line, = self._simulation_ax.plot([], [], lw=3)
        self._circle = plt.Circle((0, 0), self._radius, color='black')
        #self._cart = plt.Rectangle()
        self.time_template = 'time = %.1fs'
        self.time_text = self._simulation_ax.text(0.05, 0.9, '', transform=self._simulation_ax.transAxes)

    def initialize_plot(self):
        """
        Initial frame of the animation, required by the FuncAnimation method
        :return: any plotted objects
        """
        self._circle.center = (0, 0.6)
        self.line.set_data([], [])
        self.time_text.set_text('')

        return self.line, self.time_text

    def plot_shapes(self, state: Tuple[float, float, float, float]):
        """
        Repeated frames of the animation, required by the FuncAnimation method
        :param state: state of the cart-pole
        :return: any plotted objects
        """
        x1, v, theta, omega = state
        y1 = 0
        x2 = x1 + self._length * np.sin(theta)
        y2 = y1 + self._length * np.cos(theta)

        self.line.set_data([x1, x2], [y1, y2])
        self._circle = plt.Circle((x2, y2), self._radius, color='black')

        return self.line

    def animate(self, state_list: List = None):
        """
        Drawing the
        :param state_list:
        :return:
        """
        ani = animation.FuncAnimation(self.fig, self.plot_shapes, frames=state_list,
                                      interval=30, blit=False, init_func=self.initialize_plot)
        plt.show()

        return ani


if __name__ == "__main__":
    render = Renderer()
    x = np.linspace(-1, 1, 50)
    v = 0
    theta = np.linspace(0, np.pi, 50)
    omega = 0
    state_list = []
    for i in range(len(x)):
        state = (x[i], v, theta[i], omega)
        state_list.append(state)

    render.animate(state_list)
