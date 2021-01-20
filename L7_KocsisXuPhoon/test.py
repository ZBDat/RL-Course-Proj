import random
import logging
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from interval import Interval

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

class Gmm2CompetitorFunc:
    """
    GMM with two Gaussian, 2D
    """
    def __init__(self, domain: Interval):
        self.domain = domain
        self.gaussian_weights = [0.5, 0.5]
        self.gaussian_means = [np.array([0, 0]), np.array([0, 0])]
        Cov = np.eye(2, dtype=float)
        self.gaussian_covariances = (Cov, Cov)
        self.gaussian1 = multivariate_normal(self.gaussian_means[0], self.gaussian_covariances[0])
        self.gaussian2 = multivariate_normal(self.gaussian_means[1], self.gaussian_covariances[1])
        self.sample_cache = []
        self.activation_cache = []

    def update_gaussians(self, x, y):
        self.sample_cache.append((x, y))
        # E step, calculate posterior based on new data
        NormalConst = self.gaussian_weights[0] * self.gaussian1.pdf([x, y]) + self.gaussian_weights[1] * self.gaussian2.pdf([x, y])
        activation1 = self.gaussian_weights[0] * self.gaussian1.pdf([x, y]) / NormalConst
        activation2 = self.gaussian_weights[1] * self.gaussian2.pdf([x, y]) / NormalConst
        self.activation_cache.append((activation1, activation2))

        # M step, update the parameters
        # weights for two gaussian
        self.gaussian_weights[0] = 1 / len(self.sample_cache) * sum(pair[0] for pair in self.activation_cache)
        self.gaussian_weights[1] = 1 / len(self.sample_cache) * sum(pair[1] for pair in self.activation_cache)
        # means for two gaussian


class FunctionApproximator(ABC):
    @abstractmethod
    def update(self, x, y) -> float:
        pass

    @abstractmethod
    def estimate_max(self, x):
        pass


class CFAGMMApproximator(FunctionApproximator):

    def __init__(self, init_domain=Interval(-5, 5, closed=True), errorThreshold=0.01, sampleNumThreshold=10, a=0.001, b=10):
        self.errorThreshold = errorThreshold
        self.sampleNumThreshold = sampleNumThreshold
        self.a = a
        self.b = b

        self.competitors = [Gmm2CompetitorFunc(domain=init_domain), ]
        self.sample_number = 0
        self.active_competitors = []

    def query(self, x):
        ...

    def update(self, x, y) -> float:
        ...

    def estimate_max(self, x):
        ...


def CompetitorFuncApprox(num_iters=500):

    # variables
    goalFunc = np.sin
    input_interval = [-5, 5]
    input_step = 0.1
    mse_threshold = 5e-3

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
                for competitor in approximator.competitors:
                    if x_t in competitor.domain:
                        approximator.active_competitors.append(competitor)
                        competitor.update_gaussians(x_t, y_t)

                approximator.update(x_t, y_t)

            # Estimate the MSE
            predictions = [approximator.query(x_t) for x_t in input_values_forth]
            y_pred = np.array([prediction[0] for prediction in predictions])
            y_std = np.array([prediction[1] for prediction in predictions])

            mse = np.mean(np.square(output_values_forth - y_pred))
            mse_values.append(mse)

        epoch_count += 1
        print(f"\r Epoch: {epoch_count}, MSE: {mse}")
        if mse <= mse_threshold:
            break
