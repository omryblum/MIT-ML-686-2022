import numpy as np
import matplotlib.pyplot as plt
datapoints = np.array([[1, 1], [2, 3], [3, 4], [-0.5, -0.5], [-1, -2], [-2, -3], [-3, -4], [-4, -3]])
labels = np.array([1, 1, 1, -1, -1, -1, -1, -1])

rotation_m = np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)],[np.sin(np.pi/3), np.cos(np.pi/3)]])

datapoints = (rotation_m @ datapoints.T).T

theta = [0, 0]
theta_z = 0

T = 100

for iteration in range(T):
    for index in range(datapoints.shape[0]):
        hfun = labels[index] * (theta @ datapoints[index] + theta_z)
        if hfun <= 1e-10:
            theta += labels[index] * datapoints[index]
            theta_z += labels[index]

x_line = np.linspace(-4,4,10)
y_line = - x_line * (theta[0]/theta[1]) - theta_z/theta[1]

fig, ax = plt.subplots()
plt.scatter(datapoints[labels == 1, 0], datapoints[labels == 1, 1], c='r')
plt.scatter(datapoints[labels == -1, 0], datapoints[labels == -1, 1], c='b')
plt.plot(x_line, y_line, 'black')
plt.xlabel('X'), plt.ylabel('Y'), plt.grid()
ax.set_aspect('equal', 'box')
plt.title('theta: {theta}, theta 0: {theta_z}'.format(theta=theta, theta_z=theta_z))
plt.show()