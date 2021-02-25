import numpy as np
import theano
import theano.tensor as T
from CartPole.Environment import CartPoleEnvironment

env = CartPoleEnvironment(action_interval=0.01)
env.reset()
np.random.seed(42)
J_hist = []


class Dynamics:

    def __init__(self, dt, min_bounds=-10.0, max_bounds=10.0, mc=0.5, mp=0.5, l=1.0, g=9.80665, b=0.1, **kwargs):

        self._x = x = T.dvector("x")
        self._u = u = T.dvector("u")

        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        self.state_size = 5
        self.action_size = 1

        def f(x, u):
            """
            The system dynamics in state space representation
            :param x: system state, use sinus and cosinus to avoid periodical angle
            :param u: force F
            :return: the ODE of the system dynamics in theano symbolic form
            """
            x_ = x[..., 0]
            x_dot = x[..., 1]
            sin_theta = x[..., 2]
            cos_theta = x[..., 3]
            theta_dot = x[..., 4]
            F = u[..., 0]

            # Dynamical equation, in Deisenroth, M. P. (2010)
            numerator1 = -3*mp*l*theta_dot**2*cos_theta*sin_theta - 6*(mp+mc)*g*sin_theta - 6*(F-b*x_dot)*cos_theta
            denominator1 = 4*l*(mp+mc) - 3*mp*l*cos_theta**2
            theta_dot_dot = numerator1 / denominator1  # angular acceleration

            numerator2 = 2*mp*l*theta_dot**2*sin_theta + 3*mp*g*sin_theta*cos_theta + 4*F - 4*b*x_dot
            denominator2 = denominator1 / l
            x_dot_dot = numerator2 / denominator2  # linear acceleration

            theta = T.arctan2(sin_theta, cos_theta)
            theta += theta_dot * dt

            # using the augmented state, to be compatible with the Reward matrix
            return T.stack([
                x_ + x_dot * dt,
                x_dot + x_dot_dot * dt,
                T.sin(theta),
                T.cos(theta),
                theta_dot + theta_dot_dot * dt,
            ]).T

        inputs = [x, u]
        self._dx = f(x, u)
        self.f = theano.function(inputs, self._dx, name="f", on_unused_input='ignore', **kwargs)

        J_x, J_u = theano.gradient.jacobian(expression=f(x, u), wrt=inputs, disconnected_inputs='ignore')
        self.f_x = theano.function(inputs, J_x, name="f_x", on_unused_input='ignore', **kwargs)
        self.f_u = theano.function(inputs, J_u, name="f_u", on_unused_input='ignore', **kwargs)

    @classmethod
    def augment_state(cls, state):

        x = state[..., 0].reshape(-1, 1)
        x_dot = state[..., 1].reshape(-1, 1)
        theta = state[..., 2].reshape(-1, 1)
        theta_dot = state[..., 3].reshape(-1, 1)

        return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):

        x = state[..., 0].reshape(-1, 1)
        x_dot = state[..., 1].reshape(-1, 1)
        sin_theta = state[..., 2].reshape(-1, 1)
        cos_theta = state[..., 3].reshape(-1, 1)
        theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])


class Cost:
    """
    The cost of ilqr is in quadratic form.
    """
    def __init__(self, Q, R, Q_terminal, x_goal=None, u_goal=None):
        self.Q = np.array(Q)
        self.R = np.array(R)

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if x_goal is None:
            self.x_goal = np.zeros(Q.shape[0])
        else:
            self.x_goal = np.array(x_goal)

        if u_goal is None:
            self.u_goal = np.zeros(R.shape[0])
        else:
            self.u_goal = np.array(u_goal)

        # derivatives of quadratic forms
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

    def l(self, x, terminal=False):

        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_goal

        if terminal:
            return x_diff.T.dot(Q).dot(x_diff)

        u_diff = u - self.u_goal
        return x_diff.T.dot(Q).dot(x_diff) + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, terminal=False):
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_goal

        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, terminal=False):
        if terminal:
            return np.zeros_like(self.u_goal)

        u_diff = u - self.u_goal
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, terminal=False):
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, terminal=False):
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, terminal=False):
        if terminal:
            return np.zeros_like(self.R)
        return self._R_plus_R_T


def on_iteration(dynamics: Dynamics, iteration_count, xs, J_opt, accepted, converged):
    """
    Print the information during each iteration
    :param dynamics:
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


def backward_pass(dynamics: Dynamics, cost: Cost, x, u):

    state_size = dynamics.state_size
    action_size = dynamics.action_size
    horizon = u.shape[0]

    l = np.zeros(horizon + 1)
    l_x = np.zeros((horizon + 1, state_size))
    l_u = np.zeros((horizon, action_size))
    l_xx = np.zeros((horizon + 1, state_size, state_size))
    l_ux = np.zeros((horizon, action_size, state_size))
    l_uu = np.zeros((horizon, action_size, action_size))

    f_x = np.zeros((horizon, state_size, state_size))
    f_u = np.zeros((horizon, state_size, action_size))

    for i in range(horizon):

        f_x[i] = dynamics.f_x(x[i], u[i])
        f_u[i] = dynamics.f_u(x[i], u[i])

        l[i] = cost.l(x[i], terminal=False)
        l_x[i] = cost.l_x(x[i], u[i], terminal=False)
        l_u[i] = cost.l_u(x[i], u[i], terminal=False)
        l_xx[i] = cost.l_xx(x[i], u[i], terminal=False)
        l_ux[i] = cost.l_ux(x[i], u[i], terminal=False)
        l_uu[i] = cost.l_uu(x[i], u[i], terminal=False)

    l[-1] = cost.l(x[-1], terminal=True)
    l_x[-1] = cost.l_x(x[-1], None, terminal=True)
    l_xx[-1] = cost.l_xx(x[-1], None, terminal=True)

    V_x = l_x[-1]
    V_xx = l_xx[-1]

    k = np.zeros((horizon, dynamics.action_size))
    K = np.zeros((horizon, dynamics.action_size, dynamics.state_size))

    for i in range(horizon - 1, -1, -1):
        # calculate the Q matrices, use regularization
        Q_x = l_x[i] + f_x[i].T.dot(V_x)
        Q_u = l_u[i] + f_u[i].T.dot(V_x)
        Q_xx = l_xx[i] + f_x[i].T.dot(V_xx).dot(f_x[i])
        Q_xu = l_ux[i] + f_u[i].T.dot(V_xx).dot(f_x[i])
        Q_uu = l_uu[i] + f_u[i].T.dot(V_xx).dot(f_u[i])

        # solve for K ad k
        k[i] = -np.linalg.inv(Q_uu).dot(Q_u)
        K[i] = -np.linalg.inv(Q_uu).dot(Q_xu)

        V_x = Q_x - K[i].T.dot(Q_uu).dot(k[i])
        V_xx = Q_xx - K[i].T.dot(Q_uu).dot(K[i])

    return np.array(k), np.array(K)


def forward_pass(dynamics, cost, x, u, k, K, alpha):
    x_approx = np.zeros_like(x)
    u_approx = np.zeros_like(u)
    x_approx[0] = x[0].copy()

    horizon = np.shape(u)[0]

    for i in range(horizon):
        # action and state given by k and K. apply action to have next state
        u_approx[i] = u[i] + alpha * k[i] + K[i].dot(x_approx[i] - x[i])
        x_approx[i + 1] = dynamics.f(x_approx[i], u_approx[i])

    J = map(lambda args: cost.l(*args), zip(x_approx[:-1], u_approx, range(horizon)))
    J = sum(J) + cost.l(x[-1], u=None, terminal=True)

    return x_approx, u_approx, J


def ilqr(env, cost: Cost, dynamics, num_episodes, num_iterations, horizon, a=0.85, b=1.25, conv_threshold = 1e-6, on_iteration=on_iteration):
    """
    Th ilqr controller using theano as gradient calculator
    :param b: learning rate param.
    :param a: learning rate param.
    :param env:
    :param J: cost
    :param dynamics:
    :param num_episodes:
    :param num_iterations:
    :param horizon:
    :return:
    """
    state_size = dynamics.state_size
    action_size = dynamics.action_size
    alphas = 1.1**(-np.arange(10)**2)

    # initialize
    u = np.random.uniform(dynamics.min_bounds, dynamics.max_bounds, [horizon, action_size])
    for i in range(horizon):
        env.step(u[i][0])

    x = np.array(env.state_list)
    x = dynamics.augment_state(x)

    converged = False

    J_opt = 5000
    for e in range(num_episodes):
        accepted = False

        # Backward pass.
        k, K = backward_pass(dynamics, cost, x, u)

        # Backtracking line search.
        for alpha in alphas:
            x_approx, u_approx, J = forward_pass(dynamics, cost, x, u, k, K, alpha)

            if J < J_opt:
                if np.abs((J_opt - J) / J_opt) < conv_threshold:
                    converged = True

                u = u_approx

                env.clear()
                env.reset()
                for i in range(horizon):
                    env.step(u[i][0])

                x = np.array(env.state_list)
                x = dynamics.augment_state(x)

                J_opt = J

                # Accept this.
                accepted = True
                break

        if on_iteration:
            on_iteration(dynamics, e, x, J_opt, accepted, converged)

        if converged:
            break

    return x, u


if __name__ == "__main__":
    env = CartPoleEnvironment(action_interval=0.01)
    env.reset()

    dt = 0.01
    pole_length = 0.6

    np.random.seed(42)
    J_hist = []

    dynamics = Dynamics(dt=0.01)

    Q = np.eye(dynamics.state_size)
    Q[0, 0] = 1.0
    Q[1, 1] = Q[4, 4] = 0.0
    Q[0, 2] = Q[2, 0] = pole_length
    Q[2, 2] = Q[3, 3] = pole_length**2

    Q_terminal = 100 * np.eye(dynamics.state_size)

    # penalty for the input. Since not included in the reward, set a small value
    R = np.array([[0.01]])

    # The cost function J
    x_goal = dynamics.augment_state(np.array([0.0, 0.0, 0.0, 0.0]))
    cost = Cost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

    x, u = ilqr(env, cost, dynamics, num_episodes=200, num_iterations=150, horizon=10, a=0.85, b=1.25, on_iteration=on_iteration)
