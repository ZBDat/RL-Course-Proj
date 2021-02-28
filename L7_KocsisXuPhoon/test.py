import random
import logging
from scipy.stats import multivariate_normal, chi2
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import interval

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse
import numpy as np

matplotlib.use('TkAgg')

logger = logging.getLogger('RLFR')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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


class GMMApproximator(FunctionApproximator):
    def __init__(self, domain: interval.Interval, input_dim=1, error_threshold=1e-3, density_threshold=0.1, size_dimension=np.array([10, 2]), a=0.85, b=1.25):
        """Initialize the GMM with 2 Gaussian"""
        self.domain = domain
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
        self.gaussian_covariances = np.array([np.eye(self.D), np.eye(self.D)])

        self.gaussian1 = multivariate_normal(mean=self.gaussian_means[0], cov=self.gaussian_covariances[0])
        self.gaussian2 = multivariate_normal(mean=self.gaussian_means[1], cov=self.gaussian_covariances[1])

        self.sum_zero_order = np.array([1, 1])  # [1]_t
        self.sum_first_order = np.zeros((2, self.D))  # [z]_t
        self.sum_second_order = np.zeros((2, self.D, self.D)) # [z*z.T]_t

        self.probs_cache = None
        self.pos_cache = None

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
        """mu, std = self.query(x)
        approx_error = (y - mu) ** 2

        if approx_error >= self.error_threshold:
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

        self.gaussian1 = multivariate_normal(mean=self.gaussian_means[0], cov=self.gaussian_covariances[0])
        self.gaussian2 = multivariate_normal(mean=self.gaussian_means[1], cov=self.gaussian_covariances[1])

        self.probs_cache = None
        self.pos_cache = None

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


class CFAGMMApproximator:

    def __init__(self, init_domain=interval.Interval(-5, 5, closed=True), errorThreshold=0.01, sampleNumThreshold=5):
        self.errorThreshold = errorThreshold
        self.sampleNumThreshold = sampleNumThreshold

        self.competitors = [GMMApproximator(domain=init_domain), ]

    def get_active(self, x):
        """
        find active domains that include x
        :param x:
        :return:
        """
        active_competitors = []
        for competitor in self.competitors:
            if x in competitor.domain:
                active_competitors.append(competitor)

        return active_competitors

    def get_winner(self, x):
        """
        find the winner at the point x
        :param x: input independent variable
        :return: winner competitor
        """
        active_competitors = self.get_active(x)
        relevanceFuncs = []

        for competitor in active_competitors:

            mean, variance = competitor.query([x])
            p = competitor.gaussian_weights.dot(np.array([competitor.gaussian1.pdf(np.array([x])),
                                                          competitor.gaussian2.pdf(np.array([x]))]))

            sampleNumberEstimate = competitor.num_samples * competitor.volume_eps * p
            relevance = chi2.ppf(0.90, sampleNumberEstimate) / (sampleNumberEstimate * variance)

            relevanceFuncs.append(relevance)

        winnerCompetitor = active_competitors[relevanceFuncs.index(max(relevanceFuncs))]

        return winnerCompetitor

    def query(self, x):
        winnerCompetitor = self.get_winner(x)
        mean, variance = winnerCompetitor.query([x])

        return mean, variance

    def update(self, x, y):
        active_competitors = self.get_active(x)
        errors = []
        for competitor in active_competitors:
            mean, _ = competitor.query(x)
            errors.append((mean - y) ** 2)

        winnerCompetitor = self.get_winner(x)
        secondaryCompetitor = active_competitors[errors.index(min(errors))]

        # by combination
        if winnerCompetitor.domain == secondaryCompetitor.domain:
            intersect_domain = winnerCompetitor.domain
        else:
            intersect_domain = winnerCompetitor.domain & secondaryCompetitor.domain

        existing_domains = list(competitor.domain for competitor in self.competitors)
        if intersect_domain not in existing_domains:
            length = intersect_domain.upper_bound - intersect_domain.lower_bound
            new_competitor = GMMApproximator(domain=intersect_domain)

            # randomly select the initial gaussian means
            x_r1 = random.uniform(intersect_domain.lower_bound, intersect_domain.upper_bound)
            x_r2 = random.uniform(intersect_domain.lower_bound, intersect_domain.upper_bound)
            mean1, variance1 = self.query(x_r1)
            mean2, variance2 = self.query(x_r2)

            new_competitor.gaussian_means = np.array([[x_r1, mean1], [x_r2, mean2]])
            new_competitor.gaussian_covariances = np.array([np.diag(np.array([0.5 * length, variance1])),
                                                            np.diag(np.array([0.5 * length, variance2]))])

            self.competitors.append(new_competitor)

        # by splitting
        else:
            new_domains = self.split_domain(winnerCompetitor.domain)
            self.competitors.append(GMMApproximator(domain=new_domains[0]))
            self.competitors.append(GMMApproximator(domain=new_domains[1]))
            self.competitors.append(GMMApproximator(domain=new_domains[2]))

        return None


    @staticmethod
    def split_domain(domain: interval.Interval):
        length = domain.upper_bound - domain.lower_bound
        a = length/4
        new_domain1 = interval.Interval(domain.lower_bound, domain.lower_bound + a, closed=True)
        new_domain2 = interval.Interval(domain.lower_bound + a, domain.upper_bound - a, closed=True)
        new_domain3 = interval.Interval(domain.upper_bound - a, domain.upper_bound, closed=True)

        return new_domain1, new_domain2, new_domain3

    def eliminate_competitor(self):
        ...


def CompetitorFuncApprox(num_epochs=100):

    # variables
    goalFunc = np.sin
    input_interval = [-5, 5]
    input_step = 0.1
    it_red = 50
    mse_threshold = 0.01

    input_values_forth = np.arange(input_interval[0], input_interval[1] + input_step, input_step)
    output_values_forth = [goalFunc(x_t) for x_t in input_values_forth]

    input_values_back = np.flip(input_values_forth)
    output_values_back = np.flip(output_values_forth)

    epoch_swap_samples = [(input_values_back, output_values_back),
                          (input_values_forth, output_values_forth)]

    # initialize the approximator
    approximator = CFAGMMApproximator()

    # Training
    epoch_count = 0
    mse_values = []
    mse = None
    y_pred = None
    y_std = None
    while True:
        for input_swap_samples, output_swap_samples in epoch_swap_samples:
            # get observation (x, y)
            for x_t, y_t in zip(input_swap_samples, output_swap_samples):
                # get active competitor
                active_competitors = []
                relevanceFuncs = []
                sampleNumberEstimates = []
                for competitor in approximator.competitors:
                    if x_t in competitor.domain:
                        active_competitors.append(competitor)
                        competitor.update([x_t], y_t)

                        mean, variance = competitor.query([x_t])
                        p = competitor.gaussian_weights.dot(np.array([competitor.gaussian1.pdf(np.array([x_t, y_t])),
                                                                      competitor.gaussian2.pdf(np.array([x_t, y_t]))]))

                        sampleNumberEstimate = competitor.num_samples * competitor.volume_eps * p
                        sampleNumberEstimates.append(sampleNumberEstimate)

                        # calculate the relevance function
                        relevance = chi2.ppf(0.90, sampleNumberEstimate) / (sampleNumberEstimate * variance)
                        active_competitors.append(competitor)
                        relevanceFuncs.append(relevance)

                # get minimum number of samples
                min_sample_number = min(sampleNumberEstimates)
                # select winner
                winnerCompetitor = active_competitors[relevanceFuncs.index(max(relevanceFuncs))]
                # calculate approximation error
                w, _ = winnerCompetitor.query([x_t])
                error = (y_t - w) ** 2
            
                if error > approximator.errorThreshold and min_sample_number > approximator.sampleNumThreshold:
                    # generate new competitors
                    approximator.update(x_t, y_t)

            # Estimate the MSE
            predictions = [approximator.query(x_t) for x_t in input_values_forth]
            y_pred = np.array([prediction[0] for prediction in predictions])
            y_std = np.array([prediction[1] for prediction in predictions])

            mse = np.mean(np.square(output_values_forth - y_pred))
            mse_values.append(mse)

        epoch_count += 1
        """if epoch_count % it_red == 1:
            approximator.eliminate_competitor()"""

        print(f"\r Epoch: {epoch_count}, MSE: {mse}")
        if mse <= mse_threshold or epoch_count == num_epochs:
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


if __name__ == "__main__":
    set_seed(0)

    CompetitorFuncApprox()
