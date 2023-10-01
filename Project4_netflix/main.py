import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K_range, seed_range = range(1, 5), range(0, 5)
K = K_range[3]
for seed in seed_range:
    mixture, post = common.init(X, K, seed)

    mixture, post, cost = kmeans.run(X, mixture, post)
    print("K = {K}, seed = {seed}, cost = {cost}".format(K=K, seed=seed, cost=cost))

    # title = "Kmeans using K: {K}, seed: {seed}".format(K=K, seed=seed)
    # common.plot(X, mixture, post, title)
