import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
#Test File to test imported functions and dimensions


dt = 0.1
q_c = 0.1
Q = np.array([[(q_c * dt ** 3) / 3, (q_c * dt ** 2) / 2],
             [(q_c * dt ** 2) / 2, q_c * dt]]) 


def h(x):
    return np.array([[np.sin(x[0, 0])]])

m_0 = np.array([[1.5, 0]]).T
P_0 = Q
#convert m_0 to 1D
m = m_0.T[0]

points = MerweScaledSigmaPoints(2, alpha=0.01, beta=2, kappa=0)
sigmas = points.sigma_points(m, P_0)
x_1 = points.Wc
y_1 = points.Wm

x = np.array([[1, 1]]).T
arr = np.array([x for _ in range(5)])
arr_T = arr.T
res = x_1 @ arr.T

print("Done")