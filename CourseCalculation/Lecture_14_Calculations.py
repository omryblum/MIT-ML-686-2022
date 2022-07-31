import numpy as np

def gaussian(x, mu, var):
    p = 1/(2 * np.pi * var)**(x.size/2) * np.exp(-np.linalg.norm(mu-x)**2 / (2 * var))
    return p

# problem parameters
p = np.array([0.5, 0.5])
mu = np.array([-3, 2])
var = np.array([4, 4])
x = np.array([0.2, -0.9, -1, 1.2, 1.8])

# Calculation of E step
p1 = np.array([p[0] * gaussian(xi, mu[0], var[0]) for xi in x])
p2 = np.array([p[1] * gaussian(xi, mu[1], var[1]) for xi in x])
p_x_sum = p1 + p2
p1 /= p_x_sum
p2 /= p_x_sum

print('Posterior probabilities for the 5 points created by component 1 is:', p1)

# Update step - M Step
p_stack = np.stack((p1, p2), -1)
p_cluster_sum = np.sum(p_stack, 0)

mu = x @ p_stack / p_cluster_sum
p = p_cluster_sum / p1.size
var = np.sum(p_stack * (mu.reshape(1,-1) - x.reshape(-1,1))**2, 0) / (x[0].size * p_cluster_sum)

print('Updated cluster 1 parameters: mu: ', mu[0], ', p: ', p[0], ', var: ', var[0])




