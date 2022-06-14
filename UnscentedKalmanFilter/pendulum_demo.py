import numpy as np
import matplotlib.pyplot as plt
from UKF import predict, update


#Pendulum tracking with EKF (x 2d)
dt = 0.05
q_c = 0.1
g = 9.8
sigma = 0.1

def f(x):
    return np.transpose(np.array([[x[0, 0] + x[1, 0]*dt,
                    x[1, 0] - g * np.sin(x[0, 0]) * dt]]))

def F(x):
    return np.array([[1, dt],
                     [-1 * g * np.cos(x[0, 0]) * dt, 1]])

def H(x):
    return np.array([[np.cos(x[0, 0]), 0]])

def h(x):
    return np.array([[np.sin(x[0, 0])]])


Q = np.array([[(q_c * dt ** 3) / 3, (q_c * dt ** 2) / 2],
             [(q_c * dt ** 2) / 2, q_c * dt]]) 

R = np.array([[sigma**2]])


#Simulate Pendulum
num_steps = 200
alpha, v = 1.5, 0
true_states = [np.array([[alpha, v]]).T]
for _ in range(num_steps):
    motion_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=Q)
    new_state = f(true_states[-1]) + np.array([motion_noise]).T
    true_states.append(new_state)

#Simulate Measurement
measurement_angle = [np.array([alpha])]
for i in range(num_steps):
    measurement_noise = np.random.normal(0, sigma)
    new_angle = h(true_states[i]) + measurement_noise
    measurement_angle.append(new_angle)

true_states = np.array(true_states)
new_angle = np.array(new_angle)

filtered_states = []

#start with p = Q?
m_0 = np.array([[1.5, 0]]).T
P_0 = Q
m_current = m_0.copy()
P_current = P_0.copy()

alpha=0.01
beta=2
kappa=0

for i in range(num_steps):
    predicted_m, predicted_P = predict(f, m_current.T[0], P_current, Q, alpha, beta, kappa, 2)
    y = measurement_angle[i+1]
    m_current, P_current = update(h, R, y, predicted_m, predicted_P, alpha, beta, kappa, 2)
    filtered_states.append(m_current)

filtered_states = np.array(filtered_states)
print(len(filtered_states), len(measurement_angle))

t = np.arange(0, dt*num_steps + dt, dt)
plt.plot(t, np.sin(true_states[:, 0]))
plt.scatter(t, measurement_angle)
plt.scatter(t[1:], np.sin(filtered_states[:, 0]))
plt.show()
