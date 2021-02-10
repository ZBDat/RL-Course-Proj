import numpy as np
from numpy import pi as pi
from numpy.linalg import inv

import GPy
from IPython.display import display

from Environment import CartPoleEnvironment


"""Algorithm 1 pilco
1: init: Sample controller parameters θ ∼ N (0, I).
    Apply random control signals and record data.
2: repeat:
3:      Learn probabilistic (GP) dynamics model using all data.
4:      Model-based policy search.
5:      repeat:
6:          Approximate inference for policy evaluation,
            get Jπ(θ).
7:          Gradient-based policy improvement: get dJπ(θ)/ dθ.
8:          Update parameters θ (e.g., CG or L-BFGS).
9:      until convergence; return θ∗
10:     Set π∗ ← π(θ∗).
11:     Apply π∗ to system (single trial/episode) and record data.
12: until task learned"""


class GPdynamics:
    def __init__(self):
        ...

    def estimate(self):
        ...

    def optimize(self):
        ...


def pilco():
    """
    Reinforcement learning with Gaussian Process. Running on the workbench of Cart-Pole
    :return: any results
    """
    env = CartPoleEnvironment()
    env.reset()

    pass


if __name__ == "__main__":
    pilco()
