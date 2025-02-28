import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import cm

matrix = np.random.rand(10, 2)
stepSize = 0.025

x1_col = matrix[:, 0]
x2_col = matrix[:, 1]


# Define the function
def f(x1, x2):
    return np.sin(x1) * np.cos(x2) + np.sin(0.5 * x1) * np.cos(0.5 * x2)

# Partial derivatives
def wrtX2(x1, x2):
    return (0.5 * np.sin(0.5 * x1) * np.sin(0.5 * x2) - np.sin(x1) * np.sin(x2))

def wrtX1(x1, x2):
    return 0.5 * np.cos(0.5 * x1) * np.cos(0.5 * x2) + np.cos(x1) * np.cos(x2)

# Compute gradient
def gradient(x1, x2):
    x1_prime_col, x2_prime_col = wrtX1(x1, x2), wrtX2(x1, x2)
    return np.array([x1_prime_col, x2_prime_col])  # Ensure shape (2,)

# Gradient Descent function
def gradient_descent(threshold=1e-6, max_iters=1000):
    curr_point = np.random.rand(2)  # Start from a random point
    iteration = 0
    point_list = []
    while np.linalg.norm(gradient(curr_point[0], curr_point[1])) > threshold and iteration < max_iters:
        grad = gradient(curr_point[0], curr_point[1])
        gradient_list.append(grad)
        point_list.append(curr_point)
        curr_point = curr_point - stepSize * grad  # Ensure proper shape
        iteration += 1
    point_list.append(curr_point)
    return iteration, curr_point, f(curr_point[0], curr_point[1]), point_list

# Visualizing norm of gradients.
def viz_gradient(gradient_list):
    gradient_norms = [np.linalg.norm(grad) for grad in gradient_list]
    plt.figure()
    plt.plot(gradient_norms)
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.xlabel("Iteration")
    plt.ylabel("Norm of Gradient (Log Scale)")
    plt.title("Gradient Descent - Norm of Gradient")

def plot_gradient_descent(points, levels):
    plt.figure()
    num_of_lines = len(points)
    color=iter(cm.cool(np.linspace(0,1,num_of_lines)))
    for i, p in enumerate(points):
        c = next(color)
        plt.plot(p[0], p[1], 'o', c=c)
    plt.contour(X, Y, Z, levels=levels)
    plt.xlabel("x1")  # Label for the x-axis
    plt.ylabel("x2")  # Label for the y-axis
    plt.title("Gradient Descent Path on Contour Plot") # Title for the plot

# Run Gradient Descent
gradient_list = []
iterations, opt_point, opt_value, point_list = gradient_descent()
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
levels = np.linspace(Z.min(), Z.max(), 30)

print(f"Optimal point: {opt_point}, Function value: {opt_value}")

plot_gradient_descent(point_list, levels)
viz_gradient(gradient_list)  # Add this line to visualize the gradient norms

plt.show()
