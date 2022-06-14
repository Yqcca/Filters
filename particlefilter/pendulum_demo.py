from math import sin
import numpy as np
import matplotlib.pyplot as plt
from particlefilter import nonlinear_gaussian_adaptive_resampling_particle_filter, nonlinear_gaussian_bootstrap_filter,\
    nonlinear_gaussian_resampling_particle_filter

dt = 0.1
q = 1
g = 9.81
sigma = 1/4

Q = np.array([[(q*dt**3)/3, (q*dt**2)/2],
             [(q*dt**2)/2, q*dt]])

R = np.array([[sigma**2]])


def f(x):
    y = np.zeros(x.shape)
    y[0] = x[0] + x[1]*dt
    y[1] = x[1] - g*sin(x[0])*dt
    return y


def h(x):
    y = np.zeros(1)
    y = sin(x[0])
    return y


# Simulate Motion
num_steps = 100
x_0, y_0 = 0, 0
motion_states = [np.array([x_0, y_0])]
for i in range(num_steps):
    new_state = np.random.multivariate_normal(mean=f(motion_states[-1]), cov=Q)
    motion_states.append(new_state)
motion_states = np.array(motion_states)

# Simulate Measurement
measurement_states = [np.array([x_0])]
for i in range(num_steps):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0]), cov=R)
    new_measurement = h(motion_states[i]) + measurement_noise
    measurement_states.append(new_measurement)
measurement_states = np.array(measurement_states)

'''w1, m1 = nonlinear_gaussian_adaptive_resampling_particle_filter(f, Q, h, R, 1, num_steps, measurement_states)
w2, m2 = nonlinear_gaussian_bootstrap_filter(f, Q, h, R, 1, num_steps, measurement_states)
m3 = [sin(i) for i in m1[:, 0]]
m4 = [sin(i) for i in m2[:, 0]]'''

# Plot the x, y pos of the states
t = [i*dt for i in range(num_steps+1)]
hy = [sin(i) for i in motion_states[:, 0]]
plt.figure('angle position vs time step')
plt.plot(t, hy)
plt.scatter(t, measurement_states)

plt.xlabel('Time step')
plt.ylabel('Angle position')
plt.legend(['Position', 'Measured Postion', 'Particle Filter', 'Bootstrap Filter'])
plt.title('angle position vs time step')
plt.show()
