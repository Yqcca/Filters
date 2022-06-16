from html.entities import name2codepoint
import numpy as np
import math
import matplotlib.pyplot as plt
from particlefilter import linear_gaussian_adaptive_resampling_particle_filter

dt = 0.1
q_1 = 1
q_2 = 1
sigma_1 = 1/2
sigma_2 = 1/2

A = np.array([[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

Q = np.array([[(q_1*dt**3)/3, 0, (q_1*dt**2)/2, 0],
             [0, (q_2*dt**3)/3, 0, (q_2*dt**2)/2],
             [(q_1*dt**2)/2, 0, q_1*dt, 0],
             [0, (q_2*dt**2)/2, 0, q_2*dt]])

H = np.array([[1, 0, 0, 0],
             [0, 1, 0, 0]])

R = np.array([[sigma_1**2, 0],
             [0, sigma_2**2]])

# Simulate Motion
num_steps = 100
x_0, y_0, vx_0, vy_0 = 0, 0, 0, 0
n0 = 10
n1 = 100
n2 = 1000
n3 = 5000


motion_states = [np.array([x_0, y_0, vx_0, vy_0])]
for i in range(num_steps):
    motion_noise = np.random.multivariate_normal(mean=np.array([0, 0, 0, 0]), cov=Q)
    new_state = A @ motion_states[-1] + motion_noise
    motion_states.append(new_state)
motion_states = np.array(motion_states)

# Simulate Measurement
measurement_states = [np.array([x_0, y_0])]
for i in range(num_steps):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0, 0]), cov=R)
    new_measurement = H @ motion_states[i] + measurement_noise
    measurement_states.append(new_measurement)
measurement_states = np.array(measurement_states)

w0, m0 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, n0, num_steps, measurement_states)
w1, m1 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, n1, num_steps, measurement_states)
w2, m2 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, n2, num_steps, measurement_states)
w3, m3 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, n3, num_steps, measurement_states)

# Compute average covariance in each step
def aver_covar(w, A, N):
    l_cov = np.zeros(A.shape)
    for i in range(N):
        for a in w.shape[0]:
            l_cov += w[a][i] * A
    return l_cov[0, 0]/N, l_cov[1, 1]/N
a_cov0_x, a_cov0_y = aver_covar(w0, Q, n0)
a_cov1_x, a_cov1_y = aver_covar(w1, Q, n1)
a_cov2_x, a_cov2_y = aver_covar(w2, Q, n2)
a_cov3_x, a_cov3_y = aver_covar(w3, Q, n3)

# Compute average MSE in x position and y position
def aver_mse(m, motion_states, num_steps):
    mse_x = 0
    mse_y = 0
    for i in range(num_steps):
        mse_x += (m[i, 0] - motion_states[i, 0])**2
        mse_y += (m[i, 1] - motion_states[i, 1])**2
    mse_x = mse_x / num_steps
    mse_y = mse_y / num_steps
    return mse_x, mse_y

a_mse0_x, a_mse0_y = aver_covar(m0, motion_states, n0)
a_mse1_x, a_mse1_y = aver_covar(m1, motion_states, n1)
a_mse2_x, a_mse2_y = aver_covar(m2, motion_states, n2)
a_mse3_x, a_mse3_y = aver_covar(m3, motion_states, n3)

# Plot the x, y pos of the states
plt.figure('Car Position')
plt.plot(motion_states[:, 0], motion_states[:, 1])
plt.scatter(measurement_states[:, 0], measurement_states[:, 1])
plt.plot(m0[:, 0], m0[:, 1])
plt.plot(m1[:, 0], m1[:, 1])
plt.plot(m2[:, 0], m2[:, 1])
plt.plot(m3[:, 0], m3[:, 1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('y position vs x position')
plt.legend(['Position', 'Measured Postion', 'n=10 Particle Filter', 'n=100 Particle Filter', 'n=1000 Particle Filter', 'n=5000 Particle Filter'])
plt.savefig('number_of_particle.pdf', bbox_inches='tight')

t = [i for i in range(num_steps)]
# Plot x-position
plt.figure('x position vs time step')
plt.plot(t, motion_states[1:, 0])
plt.scatter(t, measurement_states[1:, 0])
plt.plot(t, m0[:, 0])
plt.plot(t, m1[:, 0])
plt.plot(t, m2[:, 0])
plt.plot(t, m3[:, 0])
plt.xlabel('time step')
plt.ylabel('x position')
plt.title('x position vs time step')
plt.legend(['Position', 'Measured Postion', 'n=10 Particle Filter', 'n=100 Particle Filter', 'n=1000 Particle Filter', 'n=5000 Particle Filter'])
plt.savefig('x-position.pdf', bbox_inches='tight')

# Plot y-position
plt.figure('y position vs time step')
plt.plot(t, motion_states[1:, 1])
plt.scatter(t, measurement_states[1:, 1])
plt.plot(t, m0[:, 1])
plt.plot(t, m1[:, 1])
plt.plot(t, m2[:, 1])
plt.plot(t, m3[:, 1])
plt.xlabel('time step')
plt.ylabel('y position')
plt.title('y position vs time step')
plt.legend(['Position', 'Measured Postion', 'n=10 Particle Filter', 'n=100 Particle Filter', 'n=1000 Particle Filter', 'n=5000 Particle Filter'])
plt.savefig('y-position.pdf', bbox_inches='tight')

print([a_cov0_x, a_cov1_x, a_cov2_x, a_cov3_x], [a_cov0_y, a_cov1_y, a_cov2_y, a_cov3_y])
print([a_mse0_x, a_mse1_x, a_mse2_x, a_mse3_x], [a_mse0_y, a_mse1_y, a_mse2_y, a_mse3_y])
