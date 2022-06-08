import numpy as np
import matplotlib.pyplot as plt
from EKF import predict, update


#Pendulum tracking with EKF (x 2d)
dt = 0.1
q_c = 0.1
g = 9.8
sigma_1 = 0.1
sigma_2 = 0.1

def f(x):
    return np.transpose(np.array([x[0] + x[1]*dt,
                    x[1] - g * np.sin(x[0]) * dt]))

def F(x):
    return np.array([[1, dt],
                     [-1 * g * np.cos(x[0]) * dt, 1]])

def H(x):
    return np.array([np.cos(x[0]), 0])

def h(x):
    return np.array([np.sin(x[0])])


Q = np.array([[(q_c * dt ** 3) / 3, (q_c * dt ** 2) / 2],
             [(q_c * dt ** 2) / 2, q_c * dt]]) 

R = np.array([[sigma_1**2, 0],
             [0, sigma_2**2]])

m_0 = np.array([0, 0])
P_0 = np.zeros((2, 2))

#Simulate Pendulum

num_steps = 200
alpha, v = 0, 0
true_states = [np.array([alpha, v])]
for _ in range(num_steps):
    motion_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=Q)
    new_state = f(true_states[-1]) + motion_noise
    true_states.append(new_state)

#Simulate Measurement
measurement_angle = [np.array([alpha])]
for i in range(num_steps):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R)
    #dim may be wrong
    new_angle = h(true_states[i]) + measurement_noise[0]
    measurement_angle.append(new_angle)

true_states = np.array(true_states)
new_angle = np.array(new_angle)

t = np.arange(0, dt*num_steps + dt, dt)
plt.plot(t, np.sin(true_states[:, 0]))
plt.scatter(t, measurement_angle)
plt.show()

