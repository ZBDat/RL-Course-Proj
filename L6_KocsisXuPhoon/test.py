import random
import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse
import numpy as np
from numpy import pi
import statistics

matplotlib.use('TkAgg')

logger = logging.getLogger('RLFR')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

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

class VariableResolutionPartition:
    def __init__(self, decision_boundary=None, thr_n=20, thr_var=1.12):
        self.child_1: VariableResolutionPartition = None
        self.child_2: VariableResolutionPartition = None
        self.decision_boundary = decision_boundary  # (s,a)
        self.sample_number = 0
        self.q_mean = 0
        self.q_variance = 0.81
        self.state_action_dict = dict()
        # thresholds
        self.thr_n = thr_n
        self.thr_var = thr_var

    def get_value(self, state, action):
        theta, theta_dot = state
        if self.child_1 is None and self.child_2 is None:
            return self.sample_number, self.q_mean, self.q_variance

        if all(np.array([theta, theta_dot, action]) < self.decision_boundary):
            return self.child_1.get_value(state, action)
        else:
            return self.child_2.get_value(state, action)

    def update_value(self, new_sample_state, new_sample_action, new_sample_Q):
        new_theta, new_theta_dot = new_sample_state
        # No children, just take the value and update the statistics
        if self.decision_boundary is None:
            if self.child_1 is None and self.child_2 is None:
                self.sample_number += 1
                a = 0.001
                b = 10
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
        else:
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
        self.child_1: VariableResolutionPartition = VariableResolutionPartition()
        self.child_2: VariableResolutionPartition = VariableResolutionPartition()

class FunctionApproximator(ABC):
    @abstractmethod
    def query(self, x):
        pass

    @abstractmethod
    def update(self, x, y) -> float:
        pass

    @abstractmethod
    def estimate_max(self, x):
        pass

class VariableResolutionApproximator(FunctionApproximator):
    def __init__(self):
        self.data = VariableResolutionPartition()

    def query(self, state_action: Tuple[Tuple[float, float], float]) -> Tuple[float, float]:
        state, action = state_action
        # Find the specific partition
        _, mean, variance = self.data.get_value(state, action)
        return mean, variance

    def update(self, new_sample_state_action: Tuple[Tuple[float, float], float],
               new_sample_Q: float):
        new_sample_state, new_sample_action = new_sample_state_action
        # Update the binary tree
        self.data.update_value(new_sample_state, new_sample_action, new_sample_Q)

    def estimate_max(self, state_action: Tuple[Tuple[float, float], float]):
        state, action = state_action
        # get the maximum value and the corresponding action
        action_list = np.linspace(-5, 5, 20)
        q_list = []
        mean_list = []

        for action in action_list:
            mean, variance = self.query((state, action))
            q_rand = np.random.normal(mean, np.sqrt(variance))
            q_list.append(q_rand)
            mean_list.append(mean)

        best_action = action_list[q_list.index(max(q_list))]
        max_mean = max(mean_list)

        return best_action, max_mean

class GMMApproximator(FunctionApproximator):
    def __init__(self, input_dim, error_threshold, density_threshold, size_dimension, a=0.5, b=1):
        """Initialize the GMM with 1 Gaussian"""
        self.input_dim = input_dim
        self.error_threshold = error_threshold
        self.density_threshold = density_threshold
        self.a = a
        self.b = b

        self.D = self.input_dim + 1
        self.d = size_dimension
        self.volume_eps = np.prod(self.d / 10)
        self.num_samples = 0
        self.gaussian_weights = np.array([0.5, 0.5])
        self.gaussian_means = np.zeros((2, self.D))
        self.gaussian_covariances = np.eye(self.D)
        self.gaussian_covariances = np.array([self.gaussian_covariances, self.gaussian_covariances])

        self.sum_zero_order = np.array([1, 1])  # [1]_t
        self.sum_first_order = np.zeros((2, self.D))  # [z]_t
        self.sum_second_order = np.zeros((2, self.D, self.D)) # [z*z.T]_t

        self.probs_cache = None
        # self.pos_cache = None

    @property
    def number_of_gaussians(self):
        return len(self.gaussian_weights)

    def query(self, x: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array(x)
        gaussian_means_x = self.gaussian_means[:, :-1]
        gaussian_covariances_xx = self.gaussian_covariances[:, :-1, :-1]
        gaussian_covariances_xy = self.gaussian_covariances[:, :-1, -1]
        gaussian_covariances_yx = self.gaussian_covariances[:, -1, :-1]
        gaussian_covariances_yy = self.gaussian_covariances[:, -1, -1]

        beta = self.gaussian_weights * self._multivariate_pdf(x, gaussian_means_x,
                                                              gaussian_covariances_xx)
        beta = beta / np.sum(beta)

        gaussian_covariances_xx_inv = np.linalg.inv(gaussian_covariances_xx)
        gaussian_covariances_yx_xx_inv = np.sum(
            gaussian_covariances_yx[:, None, :] * gaussian_covariances_xx_inv, axis=-1)

        gaussian_means_y = self.gaussian_means[:, -1]
        distances_to_mean = x - gaussian_means_x

        means = gaussian_means_y + np.sum(gaussian_covariances_yx_xx_inv * distances_to_mean,
                                          axis=-1)
        mean = np.sum(beta * means)

        variances = gaussian_covariances_yy - np.sum(
            gaussian_covariances_yx_xx_inv * gaussian_covariances_xy, axis=-1)
        variance = np.sum(beta * (variances + np.square(means - mean)))

        return mean, variance

    def update(self, x: List[float], y: float):
        """
        Expectation-maximization
        """
        position_vector = np.array(x + [y])
        # E step - Calculate the activations
        self.probs_cache = self._get_probs(position_vector)
        self.pos_cache = position_vector
        w = self.gaussian_weights * self.probs_cache
        density = np.sum(w)
        w = w / density

        # M step - Update the parameters
        current_value_zero_order = w
        current_value_first_order = w[:, None] * position_vector[None, :]
        current_value_second_order = w[:, None, None] * np.outer(position_vector, position_vector)[None, :]

        # Time-dependent local adjusted forgetting
        local_num_of_samples = self.num_samples * density * self.volume_eps
        remind_factor = 1 - (1 - self.a) / (self.a * local_num_of_samples + self.b)

        # weight-depending forgetting -> forgets ONLY when new information is provided
        keep_previous_value_factor = np.power(remind_factor, w)
        apply_new_value_factor = (1 - keep_previous_value_factor) / (1 - remind_factor)

        self.sum_zero_order = keep_previous_value_factor * self.sum_zero_order + apply_new_value_factor * current_value_zero_order
        self.sum_first_order = keep_previous_value_factor[:, None] * self.sum_first_order \
                               + apply_new_value_factor[:,None] * current_value_first_order
        self.sum_second_order = keep_previous_value_factor[:, None, None] * self.sum_second_order \
                                + apply_new_value_factor[:, None, None] * current_value_second_order

        self.num_samples = self.num_samples + 1
        self._update_gaussians()

        # Check whether new Gaussian is required
        mu, std = self.query(x)
        approx_error = (y - mu) ** 2

        """if approx_error >= self.error_threshold:
            self.probs_cache = self._get_probs(position_vector)
            density = np.sum(self.gaussian_weights * self.probs_cache)

            if density <= self.density_threshold:
                self.generate_gaussian(position_vector)
            else:
                self.probs_cache = None"""

    def estimate_max(self, x: List[float]):
        # get the maximum value and the corresponding action
        action_list = np.linspace(-5, 5, 20)
        q_list = []
        mean_list = []

        for action in action_list:
            mean, variance= self.query([x[0], x[1], action])
            # the probability distribution for q, p(q|s,a)
            q_rand = np.random.normal(mean, np.sqrt(variance))
            q_list.append(q_rand)
            mean_list.append(mean)

        best_action = action_list[q_list.index(max(q_list))]
        max_mean = max(mean_list)

        return best_action, max_mean

    def _update_gaussians(self):
        """
        Adjust the parameters of a GMM from samples
        """
        self.gaussian_weights = self.sum_zero_order / np.sum(self.sum_zero_order)
        self.gaussian_means = self.sum_first_order / self.sum_zero_order[:, None]
        self.gaussian_covariances = self.sum_second_order / self.sum_zero_order[:, None, None] \
                                    - self.gaussian_means[:, :, None] * self.gaussian_means[:, None,
                                                                        :]
        
        # regularization covariance matrix -> prevent singularity
        w, _ = np.linalg.eig(self.gaussian_covariances)
        cov_matrix = self.gaussian_covariances[-1, :, :]
        min_w = np.amin(w)
        while min_w < 1e-6:
            reg_coef = 0.04
            var = np.trace(cov_matrix) / (self.D)
            var = max(var, 0.01)
            self.gaussian_covariances = self.gaussian_covariances + reg_coef * np.square(var) * np.eye(self.D)[None, :]
            w, _ = np.linalg.eig(self.gaussian_covariances)
            min_w = np.amin(w)
        
        self.probs_cache = None
        # self.pos_cache = None

    # def get_activations(self, position_vector: np.ndarray):
    #     w = np.array(
    #         [weight * self._multivariate_pdf(position_vector, mean, cov) for weight, mean, cov in
    #          zip(self.gaussian_weights, self.gaussian_means, self.gaussian_covariances)])
    #     w = w / np.sum(w)
    #     return w

    # def get_probs(self, position_vector: np.ndarray):
    #     if self.pos_cache is not None:
    #         return self.probs_cache

    def plot_gaussians(self, ax, facecolor='none', edgecolor='red', **kwargs):
        for weight, mean, covariance in zip(self.gaussian_weights, self.gaussian_means,
                                            self.gaussian_covariances):
            pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
            # Using a special case to obtain the eigenvalues of this
            # two-dimensionl dataset.
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse((0, 0),
                              width=ell_radius_x * 2,
                              height=ell_radius_y * 2,
                              facecolor=facecolor,
                              edgecolor=edgecolor)

            # Calculating the stdandard deviation of x from
            # the squareroot of the variance and multiplying
            # with the given number of standard deviations.
            scale_x = np.sqrt(covariance[0, 0])
            mean_x = mean[0]

            # calculating the stdandard deviation of y ...
            scale_y = np.sqrt(covariance[1, 1])
            mean_y = mean[1]

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(mean_x, mean_y)

            ellipse.set_transform(transf + ax.transData)
            ax.add_artist(ellipse)

    def generate_gaussian(self, position_vector: np.ndarray):
        """
        New Gaussian initialization
        """
        w_new = 0.95
        zero_order_value = 1

        sum_zero_density = np.sum(self.sum_zero_order * self.probs_cache)

        # Initialization of Covariance Matrix of New Gaussian
        C = 1 / np.sqrt(2 * np.pi) \
            * np.power(
            np.square((w_new / (1 - w_new)) * (sum_zero_density / zero_order_value)) * np.prod(
                np.square(self.d))
            , -1 / (2 * self.D))

        new_mean = position_vector[None, :]
        new_covariance = np.diag(np.square(C * self.d))[None, :]

        new_zero_order_value = np.array([1.0])
        new_first_order_value = position_vector[None, :]
        new_second_order_value = new_covariance + np.outer(new_mean, new_mean)[None, :]

        self.sum_zero_order = np.concatenate((self.sum_zero_order, new_zero_order_value))
        self.sum_first_order = np.concatenate((self.sum_first_order, new_first_order_value))
        self.sum_second_order = np.concatenate((self.sum_second_order, new_second_order_value))

        self._update_gaussians()

    def _get_probs(self, position_vector: np.ndarray):
        return self._multivariate_pdf(position_vector, self.gaussian_means,
                                      self.gaussian_covariances)

    # def _pdf(self, position_vector: np.ndarray):
    #     error = position_vector - self.gaussian_means
    #     quadratic_form = np.sum(np.sum(error[:, None, :] * np.linalg.inv(self.gaussian_covariances), axis=-1) * error, axis=-1)
    #     result = np.power(2 * np.pi, - self.D / 2) / np.sqrt(np.linalg.det(self.gaussian_covariances)) * np.exp(
    #         -.5 * quadratic_form) + np.finfo(float).eps
    #     return result

    def _multivariate_pdf(self, vector: np.ndarray, mean: np.ndarray, cov: np.ndarray):
        """
        Source: https://stackoverflow.com/questions/15120662/compute-probability-over-a-multivariate-normal
        :param vector:
        :param mean:
        :param cov:
        :return:
        """
        error = vector - mean
        quadratic_form = np.sum(
            np.sum(error[:, None, :] * np.linalg.inv(cov), axis=-1) * error,
            axis=-1)
        result = np.power(2 * np.pi, - self.D / 2) / np.sqrt(
            np.linalg.det(cov)) * np.exp(
            -.5 * quadratic_form) + np.finfo(float).eps
        return result

def variable_resolution_q_learning():
    env = InvertedPendulumEnvironment()

    # initialize 
    Q_value_estimate = VariableResolutionApproximator()

    state = (pi, 0)
    action = random.choice(np.linspace(-5, 5, 20))
    Q_value_estimate.data.state_action_dict[state] = list([action])

    # loop
    num_episodes = 100
    num_iterations = 500

    eps0 = 1.25
    gamma = 0.96

    accumulated_reward = []

    for e in range(num_episodes):

        if e % 10 == 0:
            print("episode", e)
        # Training phase
        # observe current state s
        env.reset()

        for i in range(num_iterations):

            # execute a and get reward, observe new state s'
            env.state = state
            next_state, reward = env.step(action)

            # estimate Q_max
            best_action, Q_max = Q_value_estimate.estimate_max((next_state, action))

            q = reward + gamma * Q_max

            # update
            Q_value_estimate.update((state, action), q)
            state = next_state

            # select an action according to the greedy policy
            action = best_action if random.random() > 1/(eps0 + 0.001 * i) else random.choice(np.linspace(-5, 5, 20))

            #if e % 10 == 0 & i % 200 == 0:
            #    print(Q_value_estimate.query(((0, 0), 0)))

        # Testing phase
        env.reset()
        test_action = random.choice(np.linspace(-5, 5, 20))
        reward = 0

        for i in range(num_iterations):

            # execute the action
            test_next_state, test_reward = env.step(test_action)

            reward += test_reward

            max_mean = -np.inf
            best_test_next_action = 0

            # select the best action based on the learned results
            action_list = np.linspace(-5, 5, 20)
            for test_next_action in action_list:
                mean, _ = Q_value_estimate.query((test_next_state, test_next_action))

                if mean > max_mean:
                    best_test_next_action = test_next_action
                    max_mean = mean

                else:
                    pass

            test_action = best_test_next_action

        accumulated_reward.append(reward)
    
    """
    # Animation
    env.reset()
    action = random.uniform(-5, 5)
    for n in range(100):
        # Apply the action
        next_state, reward = env.step(action)

        # Render the simulation if needed
        env.render()

        # Plot the rewards if needed
        env.plot_reward()

        # Sleep for the simulation time, if needed
        env.wait_for_simulation()

        # Choose the best action after the training
        max_mean = -np.inf
        best_test_next_action = 0

        action_list = np.linspace(-5, 5, 100)
        for next_action in action_list:
            mean, _ = Q_value_estimate.query((next_state, next_action))

            if mean > max_mean:
                best_next_action = next_action
                max_mean = mean

            else:
                pass

        action = best_next_action
    """

    return accumulated_reward

def gmm_q_learning():
    env = InvertedPendulumEnvironment()

    # initialize 
    Q_value_estimate = GMMApproximator(input_dim=3, error_threshold=100.0, density_threshold=1e-5, size_dimension=np.array([20, 20, 10, 50]), a=0.9, b=1)

    state = (pi, 0)
    action = random.choice(np.linspace(-5, 5, 20))

    # loop
    num_episodes = 100
    num_iterations = 500

    eps0 = 1.25
    gamma = 0.99

    accumulated_reward = []

    for e in range(num_episodes):

        if e % 1 == 0:
            print("episode", e)

        # Training phase
        # observe current state s
        env.reset()

        for i in range(num_iterations):

            # execute a and get reward, observe new state s'
            env.state = state
            next_state, reward = env.step(action)

            # estimate Q_max
            best_action, Q_max = Q_value_estimate.estimate_max([next_state[0], next_state[1], action])

            q = reward + gamma * Q_max

            # update
            Q_value_estimate.update([state[0], state[1], action], q)
            state = next_state

            # select an action according to the greedy policy
            action = best_action if random.random() > 1/(eps0 + 0.001 * i) else random.choice(np.linspace(-5, 5, 20))

            # if e % 10 == 0 & i % 200 == 0:
            #    print(Q_value_estimate.query([0, 0, 0]))

        # Testing phase
        env.reset()
        test_action = random.choice(np.linspace(-5, 5, 20))
        reward = 0

        for i in range(num_iterations):

            # execute the action
            test_next_state, test_reward = env.step(test_action)

            reward += test_reward

            max_mean = -np.inf
            best_test_next_action = 0

            # select the best action based on the learned results
            action_list = np.linspace(-5, 5, 20)
            for test_next_action in action_list:
                mean, _ = Q_value_estimate.query([test_next_state[0], test_next_state[1], test_next_action])

                if mean > max_mean:
                    best_test_next_action = test_next_action
                    max_mean = mean

                else:
                    pass

            test_action = best_test_next_action

        if e % 1 == 0:
            print(f"\r Test Episode: {e}, Number of Gaussians: {Q_value_estimate.number_of_gaussians}, reward: {reward}")

        accumulated_reward.append(reward)

    """
    # Animation
    env.reset()
    test_action = random.choice(np.linspace(-5, 5, 20))
    reward = 0
    for n in range(100):
        # execute the action
        test_next_state, test_reward = env.step(test_action)

        # Render the simulation if needed
        env.render()

        # Plot the rewards if needed
        env.plot_reward()

        reward += test_reward

        max_mean = -np.inf
        best_test_next_action = 0

        # select the best action based on the learned results
        action_list = np.linspace(-5, 5, 20)
        for test_next_action in action_list:
            mean, _ = Q_value_estimate.query([test_next_state[0], test_next_state[1], test_next_action])

            if mean > max_mean:
                best_test_next_action = test_next_action
                max_mean = mean

            else:
                pass

        test_action = best_test_next_action
    """
    
    return accumulated_reward


def exercise_1():
    """
    Function approximation of sinus using GMM
    """
    # Define variables

    func_to_approximate = np.sin
    input_interval = [-5, 5]
    input_step = 0.1
    mse_threshold = 5e-3

    input_values_forth = np.arange(input_interval[0], input_interval[1] + input_step, input_step)
    output_values_forth = [func_to_approximate(x_t) for x_t in input_values_forth]

    input_values_back = np.flip(input_values_forth)
    output_values_back = np.flip(output_values_forth)

    epoch_swap_samples = [(input_values_back, output_values_back),
                          (input_values_forth, output_values_forth)]

    # Initialize the approximator
    approximator = GMMApproximator(input_dim=1, error_threshold=1e-3, density_threshold=0.1, size_dimension=np.array([10, 2]), a=0.9, b=1)

    # Training loop
    epoch_count = 0
    mse_values = []
    mse = None
    y_pred = None
    y_std = None
    while True:
        for input_swap_samples, output_swap_samples in epoch_swap_samples:
            # Get observation
            for x_t, y_t in zip(input_swap_samples, output_swap_samples):
                # Update the approximator
                # y_t += np.random.normal(0, 0.5)
                approximator.update([x_t], y_t)

            # Estimate the MSE
            predictions = [approximator.query(x_t) for x_t in input_values_forth]
            y_pred = np.array([prediction[0] for prediction in predictions])
            y_std = np.array([prediction[1] for prediction in predictions])

            mse = np.mean(np.square(output_values_forth - y_pred))
            mse_values.append(mse)

        epoch_count += 1
        print(f"\r Epoch: {epoch_count}, MSE: {mse}, Number of Gaussians: {approximator.number_of_gaussians}")
        if mse <= mse_threshold:
            break

    print(
        f"\rFunction approximation with {approximator.__class__.__name__} finished in {epoch_count} iterations!")
    print(f"Final MSE: {mse}")

    # Plot the MSE evolution
    plt.figure("MSE evolution")
    plt.title = "Function approximation with GMM - MSE evolution"
    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.plot(mse_values)

    # Plot the Gaussians
    plt.figure("GMM Approximator")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title = "Gaussians of the GMM"
    plt.xlabel("x")
    plt.ylabel("y = sin(x)")
    plt.plot(input_values_forth, output_values_forth)
    approximator.plot_gaussians(ax)

    # Plot the approximation
    plt.figure("Function approximation with GMM")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title = "Function approximation with GMM"
    plt.xlabel("x")
    plt.ylabel("y = sin(x)")
    plt.plot(input_values_forth, y_pred)
    plt.plot(input_values_forth, output_values_forth)
    plt.fill_between(input_values_forth, y_pred - y_std, y_pred + y_std, alpha=0.1)

    plt.ioff()
    plt.show()


def exercise_2():
    set_seed(0)
    reward1 = variable_resolution_q_learning()
    set_seed(0)
    reward2 = gmm_q_learning()

    plt.figure("Inverted Pendulum Q-Learning with FA")
    plt.title = "Function approximation Q-Learning - Rewards Evolution"
    plt.xlabel("Number of Test Episodes")
    plt.ylabel("Accumulated Rewards")
    plt.plot(reward1, 'b-', label='Variable Resolution')
    plt.plot(reward2, 'r-', label='GMM')
    plt.legend(loc='best')
    plt.show()




if __name__ == "__main__":
    set_seed(0)
    # init_logger()

    # variable_resolution_q_learning()
    exercise_1()
    # exercise_2()
