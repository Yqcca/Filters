import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from math import sin

g = 9.8
dt = 0.05
q_c = 0.1
sigma = 0.1
alpha, v = 1.5, 0

def f(x, dt):

    return np.array([x[0] + x[1]*dt, x[1] - g * np.sin(x[0]) * dt])

def h(x):

    return np.array([np.sin(x[0])])

Q = np.array([[(q_c * dt ** 3) / 3, (q_c * dt ** 2) / 2],
             [(q_c * dt ** 2) / 2, q_c * dt]]) 

R = np.array([[sigma**2]])

# Simulate Motion
num_steps = 200

motion_states = [np.array([alpha, v])]
for i in range(num_steps):
    new_state = np.random.multivariate_normal(mean=f(motion_states[-1], dt), cov=Q)
    motion_states.append(new_state)
motion_states = np.array(motion_states)


# Simulate Measurement
measurement_states = [np.array([alpha])]
for i in range(num_steps):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0]), cov=R)
    new_measurement_1 = h(motion_states[i]) + measurement_noise
    new_measurement_2 = np.random.uniform(-alpha, alpha)
    new_measurement = np.random.choice([new_measurement_1, new_measurement_2])
    measurement_states.append(new_measurement)
measurement_states = np.array(measurement_states)

sigmas = MerweScaledSigmaPoints(2, alpha=0.01, beta=2, kappa=0)
ukf = UKF(dim_x = 2, dim_z = 1, dt=dt, hx=h, fx=f, points=sigmas)


ukf.R = R
ukf.Q = Q
ukf.x = np.array([alpha, v])

filtered_states = []

for measurement in measurement_states:
    ukf.predict()
    ukf.update(measurement)
    filtered_states.append(ukf.x.copy())

filtered_states = np.array(filtered_states)

# Plot the x, y pos of the states
t = [i*dt for i in range(num_steps+1)]
hy = [sin(i) for i in motion_states[:, 0]]
plt.plot(t, hy)
plt.scatter(t, measurement_states)
plt.scatter(t, np.sin(filtered_states[:, 0]))
plt.xlabel('Time step')
plt.ylabel('Angle position')
plt.legend(['Position', 'Measured Postion', 'Particle Filter', 'Bootstrap Filter'])
plt.show()