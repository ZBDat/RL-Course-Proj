import numpy as np
import random
import matplotlib.pyplot as plt


class InvertedPendulumEnvironment():
    # TODO For TBD - 1st week
    # Implement Env and visualization
    def __init__(self, pole_mass=1.0, pole_length=1.0, delta_t=0.001):
        self.gravity = 9.8
        self.delta_t = delta_t
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.angular_position_t = 0.0 # initial position (random?)
        self.angular_velocity_t = 0.0

    def step(self):
        """
        @param action: an integer from a uniform distribution in [-5,5]
        state = (angular_position_t1, angular_velocity_t1)
        @return (angular_position_t1, angular_velocity_t1), reward
        """
        action = random.choice()
        angular_velocity_t1 = ...
        angular_position_t1 = ...
        # Rescale for values out of range for the angular_position_t1
        if angular_position_t1 

        # Check angular velocity limits

        # new position and velocity
        self.angular_position_t = angular_position_t1
        self.angular_velocity_t = angular_velocity_t1
        # Reward function
        reward = 0.0 
        return [angular_position_t1, angular_velocity_t1], reward
    
    def visualization(self):




def variable_resolution_q_learning():
    """
    TODO for TBD - 2st week
    Implement the variable resolution Q-learning algorithm
    """
    # TODO: Training
    # TODO: Inference
    pass


if __name__ == "__main__":
    random.seed(0)
    # test
    variable_resolution_q_learning()
