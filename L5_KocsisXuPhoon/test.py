import random
from abc import ABC, abstractmethod
from typing import Tuple, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Ellipse

matplotlib.use('TkAgg')


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
            if all(np.array(
                    [new_theta, new_theta_dot, new_sample_action]) < self.decision_boundary):
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
        if action is not None:
            raise NotImplementedError()
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
    def __init__(self, input_dim, error_threshold, density_threshold, a=0.5, b=1):
        """Initialize the GMM with 1 Gaussian"""
        self.input_dim = input_dim
        self.error_threshold = error_threshold
        self.density_threshold = density_threshold
        self.a = a
        self.b = b

        self.D = self.input_dim + 1
        self.d = np.array([10, 2])
        self.volume_eps = np.prod(self.d / 10)
        self.num_samples = 0
        self.gaussian_weights = np.array([1])
        self.gaussian_means = np.zeros(self.D)[None, :]
        self.gaussian_covariances = np.eye(self.D)[None, :]

        self.sum_zero_order = np.array([1])  # [1]_t
        self.sum_first_order = np.zeros(self.D)[None, :]  # [z]_t
        self.sum_second_order = np.zeros((self.D, self.D))[None, :]  # [z*z.T]_t

        self.probs_cache = None
        # self.pos_cache = None

    @property
    def number_of_gaussians(self):
        return len(self.gaussian_weights)

    def query(self, x: List[float]) -> Tuple[float, float]:
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
        current_value_second_order = w[:, None, None] * np.outer(position_vector, position_vector)[
                                                        None, :]

        # Time-dependent local adjusted forgetting
        local_num_of_samples = self.num_samples * density * self.volume_eps
        remind_factor = 1 - (1 - self.a) / (self.a * local_num_of_samples + self.b)

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

        if approx_error >= self.error_threshold:
            self.probs_cache = self._get_probs(position_vector)
            density = np.sum(self.gaussian_weights * self.probs_cache)
            if density <= self.density_threshold:
                self.generate_gaussian(position_vector)
            else:
                self.probs_cache = None

    def estimate_max(self, x: List[float]):
        # return best_action, max_mean
        raise NotImplementedError()

    def _update_gaussians(self):
        self.gaussian_weights = self.sum_zero_order / np.sum(self.sum_zero_order)
        self.gaussian_means = self.sum_first_order / self.sum_zero_order[:, None]
        self.gaussian_covariances = self.sum_second_order / self.sum_zero_order[:, None, None] \
                                    - self.gaussian_means[:, :, None] * self.gaussian_means[:, None,
                                                                        :]

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
        w_new = 0.9
        zero_order_value = 1

        sum_zero_density = np.sum(self.sum_zero_order * self.probs_cache)

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
    approximator = GMMApproximator(input_dim=1, error_threshold=1e-3, density_threshold=0.1, a=0.9,
                                   b=1)

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
        print(
            f"\r Epoch: {epoch_count}, MSE: {mse}, Number of Gaussians: {approximator.number_of_gaussians}",
            end='')
        if mse <= mse_threshold:
            break

    print(
        f"\rFunction approximation with {approximator.__class__.__name__} finished in {epoch_count} iterations!")
    print(f"Final MSE: {mse}")

    # Plot the MSE evolution
    plt.figure("MSE evolution")
    plt.title = "Function approximation with GMM - MSE evolution"
    plt.xlabel("iteration")
    plt.xlabel("MSE")
    plt.plot(mse_values)

    # Plot the Gaussians
    plt.figure("GMM Approximator")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title = "Gaussians of the GMM"
    plt.xlabel("x")
    plt.xlabel("y = sin(x)")
    plt.plot(input_values_forth, output_values_forth)
    approximator.plot_gaussians(ax)

    # Plot the approcimation
    plt.figure("Function approximation with GMM")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title = "Function approximation with GMM"
    plt.xlabel("x")
    plt.xlabel("y = sin(x)")
    plt.plot(input_values_forth, y_pred)
    plt.plot(input_values_forth, output_values_forth)
    plt.fill_between(input_values_forth, y_pred - y_std, y_pred + y_std, alpha=0.1)

    plt.ioff()
    plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(0)
    # init_logger()

    # test
    # environment_simulation()
    exercise_1()
