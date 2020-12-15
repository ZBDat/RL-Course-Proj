import numpy as np
import random
import matplotlib.pyplot as plt


def gradient_descent(domain, y, start=0, ascent=False):
    # define the function
    domain = domain
    y = y

    # parameters for the GD
    a = 0.01
    b = 10
    momentum = 0.1
    mu = 0.9
    threshold = 0.001
    num_iteration = 1000

    x = start
    route = []

    # calculate gradient
    h = 0.0001
    if not ascent:
        gradient = lambda x: (y(x + h) - y(x)) / h
    else:
        gradient = lambda x: (y(x) - y(x + h)) / h

    # loop
    for i in range(num_iteration):
        alpha = 1 / (a * i + b)
        momentum = mu * momentum + gradient(x)

        x_new = x - alpha * momentum
        if x_new > np.max(domain):
            x_new = random.uniform(2, 3)
        elif x_new < np.min(domain):
            x_new = random.uniform(-1, 1)

        if abs(x_new - x) < threshold:
            break
        else:
            x = x_new
            route.append(x)

    if not ascent:
        print("xini = %s, Finished in %i iterations" % (start, i))
        print("Minimum with GD:", (x, y(x)))
        print("Real Answer:", domain[np.argmin(y(domain))], min(y(domain)))
        print("error:", domain[np.argmin(y(domain))] - x)

    else:
        print("xini = %s, Finished in %i iterations" % (start, i))
        print("Maximum with GD:", (x, y(x)))
        print("Real Answer:", domain[np.argmax(y(domain))], max(y(domain)))
        print("error:", domain[np.argmax(y(domain))] - x)

    return route


if __name__ == "__main__":
    random.seed(0)
    domain = np.linspace(-1, 3, 1000)
    y = lambda x: 0.99 * x ** 5 - 5 * x ** 4 + 4.98 * x ** 3 + 5 * x ** 2 - 6 * x - 1

    # miminum
    route_0 = gradient_descent(domain, y, start=0)
    route_1 = gradient_descent(domain, y, start=1)
    route_2 = gradient_descent(domain, y, start=2)

    # maximum
    route_00 = gradient_descent(domain, y, start=0, ascent=True)
    route_11 = gradient_descent(domain, y, start=1, ascent=True)
    route_22 = gradient_descent(domain, y, start=2, ascent=True)

    plt.figure()
    plt.plot(domain, y(domain))
    plt.plot(np.array(route_0), y(np.array(route_0)), 'r')

    plt.figure()
    plt.plot(domain, y(domain))
    plt.plot(np.array(route_1), y(np.array(route_1)), 'g')

    plt.figure()
    plt.plot(domain, y(domain))
    plt.plot(np.array(route_2), y(np.array(route_2)), 'black')

    plt.figure()
    plt.plot(domain, y(domain))
    plt.plot(np.array(route_00), y(np.array(route_00)), 'pink')

    plt.figure()
    plt.plot(domain, y(domain))
    plt.plot(np.array(route_11), y(np.array(route_11)), 'purple')

    plt.figure()
    plt.plot(domain, y(domain))
    plt.plot(np.array(route_22), y(np.array(route_22)), 'brown')

    plt.show()
