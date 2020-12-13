import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(start=0, use_true_gradient=True):
    # define the function
    domain = np.linspace(-1, 3, 1000)
    y = lambda x: 0.99 * x ** 5 - 5 * x ** 4 + 4.98 * x ** 3 + 5 * x ** 2 - 6 * x - 1
    dy = lambda x: 4.95 * x ** 4 - 20 * x ** 3 + 14.94 * x ** 2 + 10 * x - 6

    # parameters for the GD
    a = 0.01
    b = 5
    momentum = 0
    mu = 0.4
    threshold = 0.0001
    num_iteration = 1000

    x = start
    route = []

    if use_true_gradient:
        gradient = dy

    else:
        gradient = 0

    # loop
    for i in range(num_iteration):
        alpha = 0.2
        momentum = mu * momentum + gradient(x)

        x_new = x - alpha * momentum
        if abs(x_new - x) < threshold:
            break
        else:
            x = x_new
            route.append(x)

    print("Finished in %i iterations" % i)
    print("Minimum with GD:", (x, y(x)))
    print("Real Answer:", domain[np.argmin(y(domain))], min(y(domain)))
    print("error:", domain[np.argmin(y(domain))] - x)

    plt.plot(domain, y(domain))
    plt.plot(np.array(route), y(np.array(route)))
    plt.show()

    return None


if __name__ == "__main__":
    gradient_descent(start=2)
