import logging
import random
import time
from typing import Tuple, List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

matplotlib.use('TkAgg')

logger = logging.getLogger('RLFR')


def init_logger():
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s\t| %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)


import statistics

matplotlib.use('TkAgg')

logger = logging.getLogger('RLFR')


def init_logger():
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s\t| %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)


class InvertedPendulumRenderer:
    """
    Renderer object for the 1-DOF Inverted Pendulum Environment
    """

    def __init__(self, simulation_ax=None, reward_ax=None, length=1.0, radius=0.1, dt=0.1, max_timesteps=None):
        """
        Initialize new object
        :param simulation_ax: The axis which should be used for the simulation rendering. Creates a new figure if None.
        :param reward_ax: The axis which should be used for the reward rendering. Creates a new figure if None.
        :param length: The length of the robe in the simulation.
        :param radius: The radius of the mass in the simulation.
        :param dt: The delta timestep of the simulation, used only by the reward function.
        :param max_timesteps: Sets the maximum value of the time in the reward plot.
        """
        self._simulation_ax: plt.Axes = simulation_ax
        self._reward_ax: plt.Axes = reward_ax
        self._length = length
        self._radius = radius
        self._dt = dt
        self._max_timesteps = max_timesteps

        self._static_main_circle = None
        self._static_reference_line = None
        self._static_reference_mass_circle = None

        self._dynamic_pendulum_line: plt.Line2D or None = None
        self._dynamic_mass_circle: plt.Circle or None = None
        self._dynamic_state_text: plt.Text or None = None

        self._dynamic_reward_plot: plt.Line2D or None = None

        self._simulation_ready = False
        self._reward_plot_ready = False

        self._previous_plot_time = None

        plt.ion()
        plt.show()

    def __del__(self):
        """
        Destructs the object
        """
        plt.ioff()
        plt.show()

    def _initialize_simulation(self):
        """
        Initialize the simulation plot. This method draws the static objects.
        """
        self._static_main_circle = plt.Circle((0, 0), 0.025, color='black', fill=True)
        self._static_reference_mass_circle = plt.Circle((0, self._length), self._radius, color='grey', fill=False,
                                                        linestyle='--')
        self._static_reference_line = plt.Line2D([0.0, 0.0], [0.0, self._length - self._radius], color='grey',
                                                 linestyle='--')

        self._dynamic_pendulum_line = plt.Line2D([0.0], [0.0], color='black')
        self._dynamic_mass_circle = plt.Circle((np.inf, np.inf), self._radius, color='black', fill=True)
        self._dynamic_state_text = plt.Text(1.5, 1, "Pos: \nVelo:", horizontalalignment='right')

        if self._simulation_ax is None:
            self._simulation_ax = self._create_figure()

        self._simulation_ax.axis('equal')
        limit = 1.5 * self._length
        self._simulation_ax.set_xlim(-limit, limit)
        self._simulation_ax.set_ylim(-limit, limit)
        self._simulation_ax.set_axis_off()
        self._simulation_ax.set_title("1-DOF Inverted Pendulum Simulation")

        self._simulation_ax.add_artist(self._static_main_circle)
        self._simulation_ax.add_artist(self._static_reference_mass_circle)
        self._simulation_ax.add_artist(self._static_reference_line)

        self._simulation_ax.add_artist(self._dynamic_pendulum_line)
        self._simulation_ax.add_artist(self._dynamic_mass_circle)
        self._simulation_ax.add_artist(self._dynamic_state_text)

        self._simulation_ready = True

    def _initialize_reward_plot(self):
        """
        Initialize the reward plot. This method draws the static objects.
        """
        if self._reward_ax is None:
            self._reward_ax = self._create_figure()

        self._reward_ax.set_xlim(0.0, self._max_timesteps)
        self._reward_ax.set_ylim(-np.pi, 0.0)

        self._reward_ax.set_title("1-DOF Inverted Pendulum Reward")
        self._reward_ax.set_xlabel('Time [s]')
        self._reward_ax.set_ylabel('Reward [-]')

        self._dynamic_reward_plot = plt.Line2D([0.0], [0.0], color='black')

        self._reward_ax.add_artist(self._dynamic_reward_plot)

        self._reward_plot_ready = True

    @staticmethod
    def _create_figure() -> plt.Axes:
        """
        Creates a new figure
        :return: The axis of the figure
        """
        dpi = 150
        figsize = (5, 5)

        figure = plt.figure(dpi=dpi, figsize=figsize)
        ax = figure.gca()
        return ax

    def render_simulation(self, state):
        """
        Renders the state of the simulation
        :param state: The state to render
        """
        if not self._simulation_ready:
            self._initialize_simulation()

        theta, theta_dot = state

        pos_x = np.sin(theta)
        pos_y = np.cos(theta)

        self._dynamic_pendulum_line.set_data([0.0, pos_x], [0.0, pos_y])
        self._dynamic_mass_circle.set_center((pos_x, pos_y))
        self._dynamic_state_text.set_text(f"Pos: {theta: 4.4f}\nVelo: {theta_dot: 4.4f}")

        self._simulation_ax.figure.canvas.draw()
        self._simulation_ax.figure.canvas.flush_events()

    def plot_reward(self, rewards: List[float]):
        """
        Plots the state of the simulation
        :param rewards: The list of rewards to plot
        """
        if len(rewards) == 0:
            return

        if not self._reward_plot_ready:
            self._initialize_reward_plot()

        max_time = self._dt * len(rewards)

        if self._max_timesteps is None:
            self._reward_ax.set_xlim(0.0, max_time)

        self._dynamic_reward_plot.set_data(np.linspace(0.0, max_time, num=len(rewards)), rewards)

        self._reward_ax.figure.canvas.draw()
        self._reward_ax.figure.canvas.flush_events()

    def pause_until_simulation_end(self):
        if self._previous_plot_time is None:
            self._previous_plot_time = time.time()
            return

        current_time = time.time()
        next_plot_time = self._previous_plot_time + self._dt
        required_sleep_time = next_plot_time - current_time
        if required_sleep_time > 0:
            time.sleep(required_sleep_time)
            self._previous_plot_time = next_plot_time
        else:
            self._previous_plot_time = time.time()
            logger.debug(f"There is no need to wait for the simulation end")


class InvertedPendulumEnvironment:
    """
    Class for the 1-D inverted pendulum environment
    """

    class InvertedPendulumParameters:
        """
        Class for the parameters of the inverted pendulum environment
        """

        def __init__(self, mass=1.0, length=1.0, gravitational_acceleration=9.8, mu=0.01):
            """
            Initialize a new parameter object
            :param mass: The mass of the ball
            :param length: The length of the robe
            :param gravitational_acceleration: The applied gravitational force
            :param mu: The friction coefficient of the robe
            """
            self.mass = mass
            self.length = length
            self.gravitational_acceleration = gravitational_acceleration
            self.mu = mu

    def __init__(self, environment_parameters=InvertedPendulumParameters(), action_range=(-5.0, 5.0),
                 action_interval=0.1, update_interval=0.001, renderer=None):
        """
        Initialize a new environment
        :param environment_parameters: The dynamical parameters of the environment
        :param action_range: The range of the applicable actions
        :param action_interval: The time interval of the action
        :param update_interval: The time interval of the Euler approximation
        :param renderer: The renderer object for visualization
        """
        self._environment_parameters = environment_parameters
        self._action_range = action_range
        self._action_interval = action_interval
        self._update_interval = update_interval

        if renderer is None:
            renderer = InvertedPendulumRenderer(dt=action_interval)
        self._renderer = renderer

        self.state: Tuple[float, float] or None = None
        self.time: float or None = None
        self.rewards: List[float] or None = None

        self._gravitational_force_cache = self._environment_parameters.mass * \
                                          self._environment_parameters.gravitational_acceleration * \
                                          self._environment_parameters.length
        self._inertia_cache = (self._environment_parameters.mass * self._environment_parameters.length ** 2)

    def reset(self) -> Tuple[float, float]:
        """
        Resets the environment
        :return: The initial state
        """
        self.state = (pi, 0)
        self.time = 0
        self.rewards = []

        return self.state

    def step(self, action: float) -> Tuple[Tuple[float, float], float]:
        """
        Runs one step in the environment
        :return: Tuple of the next state and the reward receive by executing the action
        """
        action_min, action_max = self._action_range
        if action < action_min or action > action_max:
            logger.debug(f"Requested action {action} exceeds the action limits{self._action_range}, "
                         f"it will be clipped!")
            action = np.clip(action, action_min, action_max)

        reward = self._get_reward(action)
        next_state = self._get_next_state(action)

        self.state = next_state
        self.time += self._action_interval
        self.rewards.append(reward)
        return next_state, reward

    def _get_reward(self, action: float) -> float:
        """
        Get a reward given the current state and the applied action
        :param action: The applied action
        :return: The reward of the behaviour
        """
        # Reward is independent from the action
        reward = -abs(self.state[0])
        return reward

    def _get_next_state(self, action: float) -> Tuple[float, float]:
        """
        Calculates the next state
        :param action: The applied action
        :return: The next state
        """
        current_position, current_velocity = self.state

        remaining_time = self._action_interval
        dt = self._update_interval

        next_position, next_velocity = current_position, current_velocity

        # Euler approximation
        while remaining_time > 0:
            if dt < remaining_time:
                dt = remaining_time

            # Dynamics
            acceleration = (-self._environment_parameters.mu * current_velocity +
                            self._gravitational_force_cache * np.sin(current_position) +
                            action) / \
                           self._inertia_cache

            # Velocity approximation: theta_dot_t = theta_dot_(t - 1) + dt * theta_dotdot
            next_velocity = self._clip_velocity(current_velocity + dt * acceleration)

            # Position approximation: theta_t = theta_(t - 1) + dt * theta_dot_t + 1 / 2 * dt ^ 2 * theta_dotdot
            next_position = current_position + dt * next_velocity + 0.5 * dt ** 2 * acceleration

            current_position, current_velocity = next_position, next_velocity
            remaining_time -= dt

        next_position = self._normalize_position(current_position)
        return next_position, next_velocity

    @staticmethod
    def _clip_velocity(velocity):
        """
        Clip the velocity to the defined range
        :param velocity: The velocity to clip
        :return: The clipped velocity
        """
        velocity_min, velocity_max = -2 * np.pi, 2 * np.pi
        if velocity < velocity_min or velocity > velocity_max:
            logger.debug(f"Velocity {velocity} exceeded the limits [{velocity_min}, {velocity_max}], "
                         f"it will be clipped")
            velocity = np.clip(velocity, velocity_min, velocity_max)
        return velocity

    @staticmethod
    def _normalize_position(position):
        """
        Normalizes the position to the given range
        :param position: The position to clip
        :return: The normalized position
        """
        position = position % (2 * np.pi)
        if position > np.pi:
            position -= (2 * np.pi)
        return position

    def render(self):
        """Renders the simulation"""
        self._renderer.render_simulation(self.state)

    def plot_reward(self):
        """Plots the rewards so far"""
        self._renderer.plot_reward(self.rewards)

    def wait_for_simulation(self):
        """Waits until the simulation step ends"""
        self._renderer.pause_until_simulation_end()


def environment_simulation():
    """
    Example function for the usage of the simulation
    """
    # Create a new environment, this environment can be used over the whole training process
    environment = InvertedPendulumEnvironment()

    # Reset the environment to a random state
    initial_state = environment.reset()

    # Render the simulation if needed
    environment.render()

    # Plot the rewards if needed
    environment.plot_reward()

    # Define some actions
    action_list = np.concatenate((np.repeat(0, 200), np.repeat(5, 100), np.repeat(0, 200)))
    for action in action_list:
        # Apply the action
        state, reward = environment.step(action)

        # Render the simulation if needed
        environment.render()

        # Plot the rewards if needed
        environment.plot_reward()

        # Sleep for the simulation time, if needed
        environment.wait_for_simulation()

    print("Done")


class VariableResolution:
    def __init__(self, decision_boundary=None, thr_n=20, thr_var=1):
        self.child_1: VariableResolution = None
        self.child_2: VariableResolution = None
        self.decision_boundary = decision_boundary  # (s,a)
        self.sample_number = 0
        self.q_mean = 0
        self.q_variance = 1
        self.state_action_dict = dict()
        # thresholds
        self.thr_n = thr_n
        self.thr_var = thr_var

    def get_value(self, state, action):
        if self.child_1 is None and self.child_2 is None:
            return self.sample_number, self.q_mean, self.q_variance

        if (state, action) < self.decision_boundary:
            self.child_1.get_value(state, action)
        else:
            self.child_2.get_value(state, action)

    def update_value(self, new_sample_state, new_sample_action, new_sample_Q):
        new_theta, new_theta_dot = new_sample_state
        # No children, just take the value and update the statistics
        if self.child_1 is None and self.child_2 is None:
            self.sample_number += 1
            a = 0.001
            b = 2
            # learning rate
            alpha = 1 / (a * self.sample_number + b)
            delta = new_sample_Q - self.q_mean
            self.q_mean = self.q_mean + alpha * delta
            self.q_variance = self.q_variance + alpha * ((delta * delta) - self.q_variance)

            if new_sample_state not in self.state_action_dict:
                self.state_action_dict[new_sample_state] = []
            self.state_action_dict[new_sample_state].append(new_sample_action)

            # Splitting criterion
            if self.q_variance > self.thr_var and self.sample_number > self.thr_n:
                self.split()

        # Has children, check the decision boundary and ask the children to proceed
        if all(np.array([new_theta, new_theta_dot, new_sample_action]) < self.decision_boundary):
            self.child_1.update_value(new_sample_state, new_sample_action, new_sample_Q)
        else:
            self.child_2.update_value(new_sample_state, new_sample_action, new_sample_Q)

    def split(self, offset=0):
        # split state-action space in 2 halves along the dimension with the largest size
        action_list = list(self.state_action_dict.values())
        action_list_flatten = [item for sublist in action_list for item in sublist]
        # size of the action dimension in Q table
        size_action = max(action_list_flatten) - min(action_list_flatten)

        state_list = self.state_action_dict.keys()
        theta_list_flatten = [state[0] for state in state_list]
        theta_dot_list_flatten = [state[1] for state in state_list]
        # size of the state dimension in Q table
        size_theta = max(theta_list_flatten) - min(theta_list_flatten)
        size_theta_dot = max(theta_dot_list_flatten) - min(theta_dot_list_flatten)

        if size_action > size_theta and size_action > size_theta_dot:
            # Split along the action dim
            boundary_action = statistics.median(action_list_flatten)
            boundary_theta = np.inf
            boundary_theta_dot = np.inf
        elif size_theta > size_action and size_theta > size_theta_dot:
            # Split along the theta dim
            boundary_action = np.inf
            boundary_theta = statistics.median(theta_list_flatten)
            boundary_theta_dot = np.inf
        else:
            # Split along the theta_dot dim
            boundary_action = np.inf
            boundary_theta = np.inf
            boundary_theta_dot = statistics.median(theta_dot_list_flatten)

        # # offset
        # boundary_state = tuple([x + offset for x in boundary_state])
        # boundary_action += offset

        self.decision_boundary = np.array([boundary_theta, boundary_theta_dot, boundary_action])
        self.child_1: VariableResolution = VariableResolution()
        self.child_2: VariableResolution = VariableResolution()


class QValue:
    def __init__(self):
        self.data = VariableResolution()
        self.data.state_action_dict[(pi, 0)] = 0

    def query(self, state: Tuple[float, float], action: float) -> Tuple[float, float]:
        # Find the specific partition
        _, mean, variance = self.data.get_value(state, action)
        return mean, variance

    def update(self, new_sample_state, new_sample_action, new_sample_Q):
        # Update the binary tree
        self.data.update_value(new_sample_state, new_sample_action, new_sample_Q)

    def estimate_max(self, state: Tuple[float, float]):
        # get the maximum value and the corresponding action
        action_list = self.data.state_action_dict[state]

        if action_list:
            mean_list = []
            max_mean = 0

            for action in action_list:
                mean, _ = self.query(state, action)
                mean_list.append(mean)
                max_mean = mean if mean > max_mean else max_mean

            best_action = action_list[mean_list.index(max_mean)]

        else:
            best_action = random.uniform(-5, 5)
            max_mean, _ = self.query(state, best_action)

        return best_action, max_mean


class InvertedPendulumRenderer:
    """
    Renderer object for the 1-DOF Inverted Pendulum Environment
    """

    def __init__(self, simulation_ax=None, reward_ax=None, length=1.0, radius=0.1, dt=0.1, max_timesteps=None):
        """
        Initialize new object
        :param simulation_ax: The axis which should be used for the simulation rendering. Creates a new figure if None.
        :param reward_ax: The axis which should be used for the reward rendering. Creates a new figure if None.
        :param length: The length of the robe in the simulation.
        :param radius: The radius of the mass in the simulation.
        :param dt: The delta timestep of the simulation, used only by the reward function.
        :param max_timesteps: Sets the maximum value of the time in the reward plot.
        """
        self._simulation_ax: plt.Axes = simulation_ax
        self._reward_ax: plt.Axes = reward_ax
        self._length = length
        self._radius = radius
        self._dt = dt
        self._max_timesteps = max_timesteps

        self._static_main_circle = None
        self._static_reference_line = None
        self._static_reference_mass_circle = None

        self._dynamic_pendulum_line: plt.Line2D or None = None
        self._dynamic_mass_circle: plt.Circle or None = None
        self._dynamic_state_text: plt.Text or None = None

        self._dynamic_reward_plot: plt.Line2D or None = None

        self._simulation_ready = False
        self._reward_plot_ready = False

        self._previous_plot_time = None

        plt.ion()
        plt.show()

    def __del__(self):
        """
        Destructs the object
        """
        plt.ioff()
        plt.show()

    def _initialize_simulation(self):
        """
        Initialize the simulation plot. This method draws the static objects.
        """
        self._static_main_circle = plt.Circle((0, 0), 0.025, color='black', fill=True)
        self._static_reference_mass_circle = plt.Circle((0, self._length), self._radius, color='grey', fill=False,
                                                        linestyle='--')
        self._static_reference_line = plt.Line2D([0.0, 0.0], [0.0, self._length - self._radius], color='grey',
                                                 linestyle='--')

        self._dynamic_pendulum_line = plt.Line2D([0.0], [0.0], color='black')
        self._dynamic_mass_circle = plt.Circle((np.inf, np.inf), self._radius, color='black', fill=True)
        self._dynamic_state_text = plt.Text(1.5, 1, "Pos: \nVelo:", horizontalalignment='right')

        if self._simulation_ax is None:
            self._simulation_ax = self._create_figure()

        self._simulation_ax.axis('equal')
        limit = 1.5 * self._length
        self._simulation_ax.set_xlim(-limit, limit)
        self._simulation_ax.set_ylim(-limit, limit)
        self._simulation_ax.set_axis_off()
        self._simulation_ax.set_title("1-DOF Inverted Pendulum Simulation")

        self._simulation_ax.add_artist(self._static_main_circle)
        self._simulation_ax.add_artist(self._static_reference_mass_circle)
        self._simulation_ax.add_artist(self._static_reference_line)

        self._simulation_ax.add_artist(self._dynamic_pendulum_line)
        self._simulation_ax.add_artist(self._dynamic_mass_circle)
        self._simulation_ax.add_artist(self._dynamic_state_text)

        self._simulation_ready = True

    def _initialize_reward_plot(self):
        """
        Initialize the reward plot. This method draws the static objects.
        """
        if self._reward_ax is None:
            self._reward_ax = self._create_figure()

        self._reward_ax.set_xlim(0.0, self._max_timesteps)
        self._reward_ax.set_ylim(-np.pi, 0.0)

        self._reward_ax.set_title("1-DOF Inverted Pendulum Reward")
        self._reward_ax.set_xlabel('Time [s]')
        self._reward_ax.set_ylabel('Reward [-]')

        self._dynamic_reward_plot = plt.Line2D([0.0], [0.0], color='black')

        self._reward_ax.add_artist(self._dynamic_reward_plot)

        self._reward_plot_ready = True

    @staticmethod
    def _create_figure() -> plt.Axes:
        """
        Creates a new figure
        :return: The axis of the figure
        """
        dpi = 150
        figsize = (5, 5)

        figure = plt.figure(dpi=dpi, figsize=figsize)
        ax = figure.gca()
        return ax

    def render_simulation(self, state):
        """
        Renders the state of the simulation
        :param state: The state to render
        """
        if not self._simulation_ready:
            self._initialize_simulation()

        theta, theta_dot = state

        pos_x = np.sin(theta)
        pos_y = np.cos(theta)

        self._dynamic_pendulum_line.set_data([0.0, pos_x], [0.0, pos_y])
        self._dynamic_mass_circle.set_center((pos_x, pos_y))
        self._dynamic_state_text.set_text(f"Pos: {theta: 4.4f}\nVelo: {theta_dot: 4.4f}")

        self._simulation_ax.figure.canvas.draw()
        self._simulation_ax.figure.canvas.flush_events()

    def plot_reward(self, rewards: List[float]):
        """
        Plots the state of the simulation
        :param rewards: The list of rewards to plot
        """
        if len(rewards) == 0:
            return

        if not self._reward_plot_ready:
            self._initialize_reward_plot()

        max_time = self._dt * len(rewards)

        if self._max_timesteps is None:
            self._reward_ax.set_xlim(0.0, max_time)

        self._dynamic_reward_plot.set_data(np.linspace(0.0, max_time, num=len(rewards)), rewards)

        self._reward_ax.figure.canvas.draw()
        self._reward_ax.figure.canvas.flush_events()

    def pause_until_simulation_end(self):
        if self._previous_plot_time is None:
            self._previous_plot_time = time.time()
            return

        current_time = time.time()
        next_plot_time = self._previous_plot_time + self._dt
        required_sleep_time = next_plot_time - current_time
        if required_sleep_time > 0:
            time.sleep(required_sleep_time)
            self._previous_plot_time = next_plot_time
        else:
            self._previous_plot_time = time.time()
            logger.debug(f"There is no need to wait for the simulation end")


class InvertedPendulumEnvironment:
    """
    Class for the 1-D inverted pendulum environment
    """

    class InvertedPendulumParameters:
        """
        Class for the parameters of the inverted pendulum environment
        """

        def __init__(self, mass=1.0, length=1.0, gravitational_acceleration=9.8, mu=0.01):
            """
            Initialize a new parameter object
            :param mass: The mass of the ball
            :param length: The length of the robe
            :param gravitational_acceleration: The applied gravitational force
            :param mu: The friction coefficient of the robe
            """
            self.mass = mass
            self.length = length
            self.gravitational_acceleration = gravitational_acceleration
            self.mu = mu

    def __init__(self, environment_parameters=InvertedPendulumParameters(), action_range=(-5.0, 5.0),
                 action_interval=0.1, update_interval=0.001, renderer=None):
        """
        Initialize a new environment
        :param environment_parameters: The dynamical parameters of the environment
        :param action_range: The range of the apllicable actions
        :param action_interval: The time interval of the action
        :param update_interval: The time interval of the Euler approximation
        :param renderer: The renderer object for visualization
        """
        self._environment_parameters = environment_parameters
        self._action_range = action_range
        self._action_interval = action_interval
        self._update_interval = update_interval

        if renderer is None:
            renderer = InvertedPendulumRenderer(dt=action_interval)
        self._renderer = renderer

        self.state: Tuple[float, float] or None = None
        self.time: float or None = None
        self.rewards: List[float] or None = None

        self._gravitational_force_cache = self._environment_parameters.mass * \
                                          self._environment_parameters.gravitational_acceleration * \
                                          self._environment_parameters.length
        self._inertia_cache = (self._environment_parameters.mass * self._environment_parameters.length ** 2)

    def reset(self) -> Tuple[float, float]:
        """
        Resets the environment
        :return: The initial state
        """
        self.state = (np.random.uniform(low=-np.pi, high=np.pi),
                      np.random.uniform(low=-2 * np.pi, high=2 * np.pi))
        self.time = 0
        self.rewards = []

        return self.state

    def step(self, action: float) -> Tuple[Tuple[float, float], float]:
        """
        Runs one step in the environment
        :return: Tuple of the next state and the reward receive by executing the action
        """
        action_min, action_max = self._action_range
        if action < action_min or action > action_max:
            logger.debug(f"Requested action {action} exceeds the action limits{self._action_range}, "
                         f"it will be clipped!")
            action = np.clip(action, action_min, action_max)

        reward = self._get_reward(action)
        next_state = self._get_next_state(action)

        self.state = next_state
        self.time += self._action_interval
        self.rewards.append(reward)
        return next_state, reward

    def _get_reward(self, action: float) -> float:
        """
        Get a reward given the current state and the applied action
        :param action: The applied action
        :return: The reward of the behaviour
        """
        # Reward is independent from the action
        reward = -abs(self.state[0])
        return reward

    def _get_next_state(self, action: float) -> Tuple[float, float]:
        """
        Calculates the next state
        :param action: The applied action
        :return: The next state
        """
        current_position, current_velocity = self.state

        remaining_time = self._action_interval
        dt = self._update_interval

        next_position, next_velocity = current_position, current_velocity

        # Euler approximation
        while remaining_time > 0:
            if dt < remaining_time:
                dt = remaining_time

            # Dynamics
            acceleration = (-self._environment_parameters.mu * current_velocity +
                            self._gravitational_force_cache * np.sin(current_position) +
                            action) / \
                           self._inertia_cache

            # Velocity approximation: theta_dot_t = theta_dot_(t - 1) + dt * theta_dotdot
            next_velocity = self._clip_velocity(current_velocity + dt * acceleration)

            # Position approximation: theta_t = theta_(t - 1) + dt * theta_dot_t + 1 / 2 * dt ^ 2 * theta_dotdot
            next_position = current_position + dt * next_velocity + 0.5 * dt ** 2 * acceleration

            current_position, current_velocity = next_position, next_velocity
            remaining_time -= dt

        next_position = self._normalize_position(current_position)
        return next_position, next_velocity

    @staticmethod
    def _clip_velocity(velocity):
        """
        Clip the velocity to the defined range
        :param velocity: The velocity to clip
        :return: The clipped velocity
        """
        velocity_min, velocity_max = -2 * np.pi, 2 * np.pi
        if velocity < velocity_min or velocity > velocity_max:
            logger.debug(f"Velocity {velocity} exceeded the limits [{velocity_min}, {velocity_max}], "
                         f"it will be clipped")
            velocity = np.clip(velocity, velocity_min, velocity_max)
        return velocity

    @staticmethod
    def _normalize_position(position):
        """
        Normalizes the position to the given range
        :param position: The position to clip
        :return: The normalized position
        """
        position = position % (2 * np.pi)
        if position > np.pi:
            position -= (2 * np.pi)
        return position

    def render(self):
        """Renders the simulation"""
        self._renderer.render_simulation(self.state)

    def plot_reward(self):
        """Plots the rewards so far"""
        self._renderer.plot_reward(self.rewards)

    def wait_for_simulation(self):
        """Waits until the simulation step ends"""
        self._renderer.pause_until_simulation_end()

    @property
    def action_range(self):
        return self._action_range


def environment_simulation():
    """
    Example function for the usage of the simulation
    """
    # Create a new environment, this environment can be used over the whole training process
    environment = InvertedPendulumEnvironment()

    # Reset the environment to a random state
    initial_state = environment.reset()

    # Render the simulation if needed
    environment.render()

    # Plot the rewards if needed
    environment.plot_reward()

    # Define some actions
    action = []
    for i in range(500):
        action.append(random.uniform(-5, 5))
    action_list = np.array(action)
    for action in action_list:
        # Apply the action
        state, reward = environment.step(action)

        # Render the simulation if needed
        environment.render()

        # Plot the rewards if needed
        environment.plot_reward()

        # Sleep for the simulation time, if needed
        environment.wait_for_simulation()

    print("Done")


def variable_resolution_q_learning():
    env = InvertedPendulumEnvironment()

    # initialize
    Q_value_estimate = QValue()

    state = (pi, 0)
    action = random.uniform(-5, 5)
    Q_value_estimate.data.state_action_dict[state] = list([action])

    # loop
    num_episodes = 100
    num_iterations = 500

    eps = 0.8
    gamma = 0.9

    accumulated_reward = 0

    for e in range(num_episodes):

        # Training phase
        # observe current state s
        env.reset()

        for i in range(num_iterations):

            # execute a and get reward, observe new state s'
            env.state = state
            next_state, reward = env.step(action)

            # estimate Q_max
            if next_state in Q_value_estimate.data.state_action_dict.keys():
                best_action, Q_max = Q_value_estimate.estimate_max(state)

            else:
                best_action = random.uniform(-5, 5)
                Q_max = Q_value_estimate.query(next_state, best_action)

            q = reward + gamma * Q_max

            # update
            Q_value_estimate.update(state, action, q)
            state = next_state

            # select an action according to the exploration-exploitation strategy
            if random.random() > eps:
                action = random.uniform(-5, 5)
            else:
                action = best_action

        # Testing phase
        env.reset()
        test_action = random.uniform(-5, 5)

        for i in range(num_iterations):

            # execute the action
            test_next_state, test_reward = env.step(test_action)

            accumulated_reward += test_reward

            max_mean = 0

            # select the best action based on the learned results
            best_test_next_action = random.choice(Q_value_estimate.data.state_action_dict.values())
            for test_next_action in Q_value_estimate.data.state_action_dict.values():
                mean, _ = Q_value_estimate.query(test_next_state, test_next_action)

                if mean >= max_mean:
                    best_test_next_action = test_next_action
                else:
                    pass

            test_action = best_test_next_action


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(0)
    # init_logger()
    # test
    # environment_simulation()
    variable_resolution_q_learning()
