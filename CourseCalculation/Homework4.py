import numpy as np

def gaussian(x, mu, var):
    p = 1/(2 * np.pi * var)**(x.size/2) * np.exp(-np.linalg.norm(mu-x)**2 / (2 * var))
    return p

# 4.3
M_prob = np.array([0.5, 0.5])
M_mu = np.array([6, 7])
M_var = np.array([1, 4])
x = np.array([-1, 0, 4, 5, 6])

# E Step, deciding which point belong to which gaussian
x_prob = np.array([[gaussian(xi, M_mu[0], M_var[0]), gaussian(xi, M_mu[1], M_var[1])] for xi in x])
x_label = np.argmax(x_prob, 1)
# # Total probability
# l_prob = np.sum(np.log(np.sum(x_prob, 1)))

# M Step, change theta parameters






# 4.1
# problem parameters
x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])
mu = np.array([[-5, 2], [0, -6]])

# K Medioids L1 norm
for _ in range(10):
    x_label = np.array([np.argmin(np.sum(np.abs(mu - xi), 1)) for xi in x])

    for index_mu, _ in enumerate(mu):
        loss_per_x = np.array([np.sum(np.sum(np.abs(z_option - x[x_label == index_mu]), 1)) for z_option in x])
        mu[index_mu, :] = x[np.argmin(loss_per_x), :]

print('Calculation for L1 Norm:')
print('New center 0 is:', mu[0, :], 'with xi:', x[x_label == 0])
print('New center 1 is:', mu[1, :], 'with xi:', x[x_label == 1])

# problem parameters
x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])
mu = np.array([[-5, 2], [0, -6]])
# K Medioids L2 norm
for _ in range(10):
    x_label = np.array([np.argmin(np.sum((mu - xi)**2, 1)**0.5) for xi in x])

    for index_mu, _ in enumerate(mu):
        loss_per_x = np.array([np.sum(np.sum((z_option - x[x_label == index_mu])**2, 1)**0.5) for z_option in x])
        mu[index_mu, :] = x[np.argmin(loss_per_x), :]

print('Calculation for L2 Norm:')
print('New center 0 is:', mu[0, :], 'with xi:', x[x_label == 0])
print('New center 1 is:', mu[1, :], 'with xi:', x[x_label == 1])

# problem parameters
x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])
mu = np.array([[-5, 2], [0, -6]])
# K means L1 norm
for _ in range(10):
    x_label = np.array([np.argmin(np.sum(np.abs(mu - xi), 1)) for xi in x])

    for index_mu, _ in enumerate(mu):
        mu[index_mu, :] = np.median(x[x_label == index_mu], 0)
        # loss_per_x = np.array([np.sum(np.sum(x[x_label == index_mu], 0)) for z_option in x])
        # mu[index_mu, :] = x[np.argmin(loss_per_x), :]

print('K means Calculation for L1 Norm:')
print('New center 0 is:', mu[0, :], 'with xi:', x[x_label == 0])
print('New center 1 is:', mu[1, :], 'with xi:', x[x_label == 1])


#
# # Update step - M Step
# p_stack = np.stack((p1, p2), -1)
# p_cluster_sum = np.sum(p_stack, 0)
#
# mu = x @ p_stack / p_cluster_sum
# p = p_cluster_sum / p1.size
# var = np.sum(p_stack * (mu.reshape(1,-1) - x.reshape(-1,1))**2, 0) / (x[0].size * p_cluster_sum)
#
# print('Updated cluster 1 parameters: mu: ', mu[0], ', p: ', p[0], ', var: ', var[0])
#



