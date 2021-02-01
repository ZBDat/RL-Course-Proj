# This exercise referred to the example of ilqr package of python
# https://github.com/anassinator/ilqr
# to successfully run the script, installation of ilqr package is required.
# The dynamics of the system is setup using the dynamics module of ilqr
# that may vary from the original implementation of this project

from abc import ABC
from typing import List

from matplotlib import pyplot as plt
from matplotlib import animation

import numpy as np
from theano import tensor as T

from L9_XuPhoon.ilqr.ilqr import iLQR
from L9_XuPhoon.ilqr.ilqr.cost import QRCost
from L9_XuPhoon.ilqr.ilqr.dynamics import BatchAutoDiffDynamics, tensor_constrain, constrain


class InvertedPendulumDynamics(BatchAutoDiffDynamics, ABC):

    """Inverted pendulum auto-differentiated dynamics model."""

    def __init__(self, dt, constrain=True, min_bounds=-5.0, max_bounds=5.0, m=1.0, l=1.0, g=9.8, mu=0.01, **kwargs):
        """Constructs an InvertedPendulumDynamics model.
        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N m].
            max_bounds: Maximum bounds for action [N m].
            m: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                BatchAutoDiffDynamics constructor.
        Note:
            state: [sin(theta), cos(theta), theta']
            action: [torque]
            theta: 0 is pointing up and increasing counter-clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        def f(x, u, i):
            # Constrain action space.
            if constrain:
                u = tensor_constrain(u, min_bounds, max_bounds)

            sin_theta = x[..., 0]
            cos_theta = x[..., 1]
            theta_dot = x[..., 2]
            torque = u[..., 0]

            # Deal with angle wrap-around.
            theta = T.arctan2(sin_theta, cos_theta)

            # Define acceleration.
            theta_dot_dot = - mu / (m * l ** 2) * theta_dot + g / l * T.sin(theta + np.pi)
            theta_dot_dot += 1 / (m * l ** 2) * torque

            next_theta = theta + theta_dot * dt

            return T.stack([
                T.sin(next_theta),
                T.cos(next_theta),
                theta_dot + theta_dot_dot * dt,
            ]).T

        super(InvertedPendulumDynamics, self).__init__(f,
                                                       state_size=3,
                                                       action_size=1,
                                                       **kwargs)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).
        In this case, it converts:
            [theta, theta'] -> [sin(theta), cos(theta), theta']
        Args:
            state: State vector [reduced_state_size].
        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            theta, theta_dot = state
        else:
            theta = state[..., 0].reshape(-1, 1)
            theta_dot = state[..., 1].reshape(-1, 1)

        return np.hstack([np.sin(theta), np.cos(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.
        In this case, it converts:
            [sin(theta), cos(theta), theta'] -> [theta, theta']
        Args:
            state: Augmented state vector [state_size].
        Returns:
            Reduced state size [reduced_state_size].
        """
        if state.ndim == 1:
            sin_theta, cos_theta, theta_dot = state
        else:
            sin_theta = state[..., 0].reshape(-1, 1)
            cos_theta = state[..., 1].reshape(-1, 1)
            theta_dot = state[..., 2].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([theta, theta_dot])


class Renderer:

    def __init__(self, length=1.0, radius=0.1, x_range=(-2, 2), y_range=(-2, 2)):
        self.length = length
        self.radius = radius

        self.fig = plt.figure()
        self.fig.set_dpi(100)

        self.simulation_ax = self.fig.add_subplot(111, xlim=x_range, ylim=y_range)
        self.simulation_ax.set_title('Cart-Pole Simulation')

        self.simulation_ax.grid()

        self.patch1 = plt.Circle((0, 0), 0.1, fc='black', ec='black')
        self.patch2 = plt.Line2D((0, self.length), (0, 0), lw=1.5)

    def initialize_plot(self):
        """
        The starting frame of the animation, required by the FuncAnimation method
        :return:
        """
        self.patch1.center = (0, -0.6)
        self.patch2.set_data(np.array([0, 0]), np.array([0, 0.6]))

        self.simulation_ax.add_artist(self.patch1)
        self.simulation_ax.add_artist(self.patch2)

        return self.patch1, self.patch2,

    def plot_shapes(self, state: float):
        """
        Repeated frames of the animation, required by the FuncAnimation method
        :param state: state of the pendulum
        :return: any plotted objects
        """
        x1, y1 = (0.0, 0.0)
        x2, y2 = self.patch1.center
        theta = state
        y1 = 0
        x2 = x1 + self.length * np.sin(theta)
        y2 = y1 + self.length * np.cos(theta)
        self.patch1.center = (x2, y2)
        self.patch2.set_data(np.array([x1, x2]), np.array([y1, y2]))

        return self.patch1, self.patch2,

    def animate(self, state_list: List):
        """
        Animation. the list of state and reward should have the same length
        :param state_list: list recording the states
        :return: a FuncAnimation object
        """

        # Tweaking option: set interval higher: slow down the animation
        ani1 = animation.FuncAnimation(self.fig, self.plot_shapes, frames=state_list,
                                       interval=20, blit=True, init_func=self.initialize_plot)

        plt.show()

        return ani1,


def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    """
    Print the information during each iteration
    :param iteration_count: number of iteration
    :param xs: state
    :param us: input
    :param J_opt: optimal cost function
    :param accepted: if the solution is valid
    :param converged: if the process already converged
    :return: None
    """
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


if __name__ == "__main__":

    dt = 0.01
    pendulum_length = 1.0
    dynamics = InvertedPendulumDynamics(dt, l=pendulum_length)
    renderer = Renderer(length=pendulum_length)

    # use augmented state
    x_goal = dynamics.augment_state(np.array([0.0, 0.0]))

    # set Q matrix. entries are penalizing factors. The Q and R matrices can be customized
    Q = np.eye(dynamics.state_size)
    Q[0, 1] = Q[1, 0] = pendulum_length
    Q[0, 0] = Q[1, 1] = 2 * pendulum_length ** 2
    Q[2, 2] = 0.0
    Q_terminal = 100 * np.eye(dynamics.state_size)

    # penalty for the input
    R = np.array([[0.2]])

    # The cost function J
    cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

    N = 500  # horizon
    x0 = dynamics.augment_state(np.array([np.pi, 0.0]))  # initial state
    us_init = np.random.uniform(-1.0, 1.0, (N, dynamics.action_size))  # initial input, here input is scaled
    ilqr = iLQR(dynamics, cost, N)

    J_hist = []
    xs, us = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)

    # Reduce the state to something more reasonable.
    xs = dynamics.reduce_state(xs)

    # Constrain the actions to see what's actually applied to the system.
    us = constrain(us, dynamics.min_bounds, dynamics.max_bounds)

    t = np.arange(N) * dt
    theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
    theta_dot = xs[:, 1]

    ani = renderer.animate(state_list=theta.tolist())

    # plot the results
    fig1 = plt.figure()
    _ = plt.plot(theta, theta_dot)
    _ = plt.xlabel("theta (rad)")
    _ = plt.ylabel("theta_dot (rad/s)")
    _ = plt.title("Phase Plot")

    fig2 = plt.figure()
    _ = plt.plot(t, us)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Force (N)")
    _ = plt.title("Action path")

    fig3 = plt.figure()
    _ = plt.plot(J_hist)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Total cost")
    _ = plt.title("Total cost-to-go")

    plt.show()
