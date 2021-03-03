from matplotlib import pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from Environment import CartPoleEnvironment
from Renderer import Renderer

trajectory_cost = []


class Dynamics:

    def __init__(self, dt, u_bounds=(-10.0, 10.0), mc=0.5, mp=0.5, l=1.0, g=9.80665, b=0.1):
        self._x = x = T.dvector("x")
        self._u = u = T.dvector("u")

        self.u_min = min(u_bounds)
        self.u_max = max(u_bounds)

        self.state_size = 5
        self.action_size = 1

        def f(x, u):
            """
            The system dynamics in state space representation
            :param x: system state, use sinus and cosinus to avoid periodical angle
            :param u: force F
            :return: the ODE of the system dynamics in theano symbolic form
            """
            # squashing the input with the limits
            diff = (self.u_max - self.u_min) / 2.0
            mean = (self.u_min + self.u_max) / 2.0
            u = diff * T.tanh(u) + mean

            x_ = x[..., 0]
            x_dot = x[..., 1]
            sin_theta = x[..., 2]
            cos_theta = x[..., 3]
            theta_dot = x[..., 4]
            F = u[..., 0]

            # Dynamical equation, in Deisenroth, M. P. (2010)
            temp = (F - 0.2 * x_dot + mp * l * theta_dot**2 * sin_theta) / (mc + mp)
            numerator = g * sin_theta - cos_theta * temp
            denominator = l * (4.0 / 3.0 - mp * cos_theta**2 / (mc + mp))
            theta_dot_dot = numerator / denominator

            x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

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
        self._state = f(x, u)
        self.f = theano.function(inputs, self._state, name="f", on_unused_input='ignore')

        J_x, J_u = theano.gradient.jacobian(expression=f(x, u), wrt=inputs, disconnected_inputs='ignore')
        self.f_x = theano.function(inputs, J_x, name="f_x", on_unused_input='ignore')
        self.f_u = theano.function(inputs, J_u, name="f_u", on_unused_input='ignore')

    def augment_state(self, state):
        """
        transform a state with theta into aug-state with sin and cos
        :param state: original state
        :return: augmented state
        """
        x = state[..., 0].reshape(-1, 1)
        x_dot = state[..., 1].reshape(-1, 1)
        theta = state[..., 2].reshape(-1, 1)
        theta_dot = state[..., 3].reshape(-1, 1)

        return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

    def reduce_state(self, state):
        """
        inverse transformation
        :param state: augmented state
        :return: original state
        """
        x = state[..., 0].reshape(-1, 1)
        x_dot = state[..., 1].reshape(-1, 1)
        sin_theta = state[..., 2].reshape(-1, 1)
        cos_theta = state[..., 3].reshape(-1, 1)
        theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])


class Cost:

    def __init__(self, Q, R, Q_terminal, x_target=None):
        """
        The cost function of iLQR is in quadratic form.
        :param Q: penalty matrix for state
        :param R: penalty matrix for action
        :param Q_terminal: final state cost
        :param x_target: final state
        """
        self.Q = np.array(Q)
        self.R = np.array(R)

        self.Q_terminal = np.array(Q_terminal)

        self.x_target = np.array(x_target)
        self.u_target = np.zeros(R.shape[0])

        # derivatives of quadratic forms
        self._QQT = self.Q + self.Q.T
        self._RRT = self.R + self.R.T
        self._QQT_Terminal = self.Q_terminal + self.Q_terminal.T

    def l(self, x, u, terminal=False):

        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        d_x = x - self.x_target

        if terminal:
            return d_x.T.dot(Q).dot(d_x)
        else:
            d_u = u - self.u_target
            return d_x.T.dot(Q).dot(d_x) + d_u.T.dot(R).dot(d_u)

    def l_x(self, x, terminal=False):
        QQT = self._QQT_Terminal if terminal else self._QQT
        d_x = x - self.x_target

        return d_x.T.dot(QQT)

    def l_u(self, u, terminal=False):
        if terminal:
            return np.zeros_like(self.u_target)

        d_u = u - self.u_target
        return d_u.T.dot(self._RRT)

    def l_xx(self, terminal=False):
        return self._QQT_Terminal if terminal else self._QQT

    def l_ux(self):
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, terminal=False):
        if terminal:
            return np.zeros_like(self.R)
        return self._RRT


def iter_info(dynamics: Dynamics, episode, x, optimal_cost, accepted, converged):
    """
    Print the information during the iteration
    :param dynamics:
    :param episode: number of iteration
    :param x: state
    :param optimal_cost: optimal cost function
    :param accepted: if the solution is valid
    :param converged: if the process already converged
    :return: None
    """
    trajectory_cost.append(optimal_cost)
    info = "converged, cost" if converged else ("accepted, cost" if accepted else "failed, cost")
    if episode % 10 == 0 or converged:
        print("episode", episode, info, optimal_cost)


def ilqr(cost: Cost, dynamics: Dynamics, init_state, num_episodes, horizon,
         conv_threshold=1e-6, mu_min=1e-6, iter_info=iter_info):
    """
    The ilqr controller using theano as gradient calculator
    :param iter_info:
    :param mu_min:
    :param conv_threshold:
    :param cost: cost
    :param dynamics:
    :param num_episodes:
    :param horizon:
    :return:
    """
    env = CartPoleEnvironment(action_interval=0.1)
    env.reset()
    action_size = dynamics.action_size

    # initialize
    u = np.random.uniform(dynamics.u_min, dynamics.u_max, [horizon, action_size])
    mu = 1.0
    delta = 2.0  # regularization terms
    alphas = 1.1 ** (-np.arange(10) ** 2)  # learning rates
    converged = False

    for e in range(num_episodes):
        accepted = False

        # apply the best control so far to get the trajectory for evaluation
        state_size = dynamics.state_size
        action_size = dynamics.action_size
        horizon = u.shape[0]

        x = np.empty((horizon + 1, state_size))
        f_x = np.empty((horizon, state_size, state_size))
        f_u = np.empty((horizon, state_size, action_size))

        l = np.empty(horizon + 1)
        l_x = np.empty((horizon + 1, state_size))
        l_u = np.empty((horizon, action_size))
        l_xx = np.empty((horizon + 1, state_size, state_size))
        l_xu = np.empty((horizon, action_size, state_size))
        l_uu = np.empty((horizon, action_size, action_size))

        x[0] = init_state
        for i in range(horizon):

            _, reward, _ = env.step(u[i][0])
            x[i + 1] = dynamics.f(x[i], u[i])
            x[i + 1] = np.clip(x[i+1], -10.0,  10.0)
            f_x[i] = dynamics.f_x(x[i], u[i])
            f_u[i] = dynamics.f_u(x[i], u[i])

            l[i] = cost.l(x[i], u[i], terminal=False)
            l_x[i] = cost.l_x(x[i], terminal=False)
            l_u[i] = cost.l_u(u[i], terminal=False)
            l_xx[i] = cost.l_xx()
            l_xu[i] = cost.l_ux()
            l_uu[i] = cost.l_uu()

        l[-1] = cost.l(x[-1], None, terminal=True)
        l_x[-1] = cost.l_x(x[-1], terminal=True)
        l_xx[-1] = cost.l_xx(terminal=True)
        optimal_total_cost = l.sum()

        # Backward pass.
        V_x = l_x[-1]
        V_xx = l_xx[-1]

        k = np.empty((horizon, dynamics.action_size))
        K = np.empty((horizon, dynamics.action_size, dynamics.state_size))

        for i in range(horizon - 1, -1, -1):

            Q_x = l_x[i] + f_x[i].T.dot(V_x)
            Q_u = l_u[i] + f_u[i].T.dot(V_x)
            Q_xx = l_xx[i] + f_x[i].T.dot(V_xx).dot(f_x[i])

            # (refer to the paper) regularization methods
            reg = mu * np.eye(dynamics.state_size)
            Q_ux = l_xu[i] + f_u[i].T.dot(V_xx + reg).dot(f_x[i])
            Q_uu = l_uu[i] + f_u[i].T.dot(V_xx + reg).dot(f_u[i])

            k[i] = -np.linalg.inv(Q_uu).dot(Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)

        # line search.
        for alpha in alphas:
            # forward pass
            x_hat = np.zeros_like(x)
            u_hat = np.zeros_like(u)
            x_hat[0] = x[0].copy()

            total_cost = 0
            for i in range(horizon):
                u_hat[i] = u[i] + alpha * k[i] + K[i].dot(x_hat[i] - x[i])
                x_hat[i + 1] = dynamics.f(x_hat[i], u_hat[i])
                x_hat[i + 1] = np.clip(x_hat[i + 1], -10.0, 10.0)
                total_cost += cost.l(x_hat[i], u_hat[i])

            total_cost += cost.l(x_hat[-1], None, terminal=True)

            if total_cost < optimal_total_cost:
                # convergence judgement from Yuval Tassa 2012
                if np.abs((optimal_total_cost - total_cost) / optimal_total_cost) < conv_threshold:
                    converged = True

                optimal_total_cost = total_cost
                u = u_hat

                # regularization
                delta = min(1.0, delta) / 2.0
                mu *= delta
                if mu <= mu_min:
                    mu = 0.0

                # Accept this.
                accepted = True
                break

        env.terminate_episode()

        if iter_info:
            iter_info(dynamics, e, x, optimal_total_cost, accepted, converged)

        if converged:
            break

    return x, u, env.accumulated_reward


if __name__ == "__main__":
    dt = 0.01
    pole_length = 0.6

    np.random.seed(1)

    dynamics = Dynamics(dt=0.01)
    env = CartPoleEnvironment(action_interval=0.01)
    renderer = Renderer(x_range=(-2.0, 2.0))

    # penalty for the state. take the T^-1 matrix from Lec. 10.
    Q = np.eye(dynamics.state_size)
    Q[0, 0] = 1.0
    Q[1, 1] = Q[4, 4] = 0.0
    Q[0, 2] = Q[2, 0] = pole_length
    Q[2, 2] = Q[3, 3] = pole_length ** 2
    Q *= 0.5

    Q_terminal = 100 * np.eye(dynamics.state_size)

    # penalty for the input. Since not included in the reward, set a small value
    R = np.array([[0.001]])

    # The cost function J
    init_state = dynamics.augment_state(np.array([0.0, 0.0, np.pi, 0.0])).reshape(5)
    final_state = dynamics.augment_state(np.array([0.0, 0.0, 0.0, 0.0])).reshape(5)
    cost = Cost(Q, R, Q_terminal=Q_terminal, x_target=final_state)

    x, u, accumulated_rewards = ilqr(cost, dynamics, init_state, num_episodes=500, horizon=450, iter_info=iter_info)
    x = dynamics.reduce_state(x)

    reward_list = []
    for i in range(x.shape[1]):
       reward = env.get_reward(tuple(x[i].tolist()))
       reward_list.append(reward)

    ani1, ani2 = renderer.animate(state_list=x.tolist(), reward_list=reward_list)

    # plot the results
    fig1 = plt.figure()
    _ = plt.plot(trajectory_cost)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("cost")

    t = np.arange(280) * dt
    theta = np.unwrap(x[:, 0])  # Makes for smoother plots.
    theta_dot = x[:, 1]

    fig2 = plt.figure()
    _ = plt.plot(theta, theta_dot)
    _ = plt.xlabel("theta (rad)")
    _ = plt.ylabel("theta_dot (rad/s)")
    _ = plt.title("Phase Plot")

    plt.show()
