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
res = x_1 * arr.T
res = res.T
g = 9.8

def f(x):
    return np.transpose(np.array([[x[0, 0] + x[1, 0]*dt,
                    x[1, 0] - g * np.sin(x[0, 0]) * dt]]))

sigmas = np.array([[ 1.50000000e+00,  0.00000000e+00],
              [ 1.50002887e+00,  8.66025404e-04],
              [ 1.50000000e+00,  5.00000000e-04],
              [ 1.49997113e+00, -8.66025404e-04],
              [ 1.50000000e+00, -5.00000000e-04]])

f_sigmas = np.empty((5, 2))
for i in range(5):
    f_sigmas[i] = f(np.array([[sigmas[i, k] for k in range(2)]]).T).T

print("k")
k = points.Wc * f_sigmas.T
p_m = np.sum(points.Wc * f_sigmas.T, axis=0)
print("Done")