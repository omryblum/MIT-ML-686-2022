import numpy as np

Xcoord = np.array([[0,0],
                  [2,0],
                  [1,1],
                  [0,2],
                  [3,3],
                  [4,1],
                  [5,2],
                  [1,4],
                  [4,4],
                  [5,5]])

label = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])

mistakes = np.array([1,65,11,31,72,30,0,21,4,15])

# Calculations
theta_zero = np.dot(label, mistakes)
X_altered = np.array([Xcoord[:, 0]**2, np.sqrt(2)*Xcoord[:, 0]*Xcoord[:, 1], Xcoord[:, 1]**2])

theta = X_altered @ (mistakes * label)

# Check labels
all_true = (np.sign(theta @ X_altered + theta_zero) * label).all()


