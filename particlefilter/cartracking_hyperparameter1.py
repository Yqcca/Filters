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
n0 = 100
n1 = 300
n2 = 500
n3 = 1000
n4 = 1500
n5 = 3000


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
w4, m4 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, n4, num_steps, measurement_states)
w5, m5 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, n5, num_steps, measurement_states)

# Compute avg MSE in position
def avg_mse(m, motion_states, num_steps):
    mse_x = 0
    mse_y = 0
    for i in range(num_steps-1):
        mse_x += (m[i, 0] - motion_states[i, 0])**2
        mse_y += (m[i, 1] - motion_states[i, 1])**2
    return (mse_x + mse_y)/(2*num_steps)

a_mse0= avg_mse(m0, motion_states, num_steps)
a_mse1= avg_mse(m1, motion_states, num_steps)
a_mse2= avg_mse(m2, motion_states, num_steps)
a_mse3= avg_mse(m3, motion_states, num_steps)
a_mse4= avg_mse(m4, motion_states, num_steps)
a_mse5= avg_mse(m5, motion_states, num_steps)

# Plot the x, y pos of the states
plt.figure('Car Position')
plt.plot(motion_states[:, 0], motion_states[:, 1])
plt.scatter(measurement_states[:, 0], measurement_states[:, 1])
plt.plot(m0[:, 0], m0[:, 1])
plt.plot(m1[:, 0], m1[:, 1])
plt.plot(m2[:, 0], m2[:, 1])
plt.plot(m3[:, 0], m3[:, 1])
plt.plot(m4[:, 0], m4[:, 1])
plt.plot(m5[:, 0], m5[:, 1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('y position vs x position')
plt.legend(['Position', 'Measured Postion', 'N=100 Particle Filter', 'N=300 Particle Filter', 'N=500 Particle Filter', 'N=1000 Particle Filter', 'N=1500 Particle Filter', 'N=3000 Particle Filter'])
plt.savefig('number_of_particle1.pdf', bbox_inches='tight')

t = [i for i in range(num_steps)]
# Plot x-position
plt.figure('x position vs time step')
plt.plot(t, motion_states[1:, 0])
plt.scatter(t, measurement_states[1:, 0])
plt.plot(t, m0[:, 0])
plt.plot(t, m1[:, 0])
plt.plot(t, m2[:, 0])
plt.plot(t, m3[:, 0])
plt.plot(t, m4[:, 0])
plt.plot(t, m5[:, 0])
plt.xlabel('Time step')
plt.ylabel('x position')
plt.title('x position vs time step')
plt.legend(['Position', 'Measured Postion', 'N=100 Particle Filter', 'N=300 Particle Filter', 'N=500 Particle Filter', 'N=1000 Particle Filter', 'N=1500 Particle Filter', 'N=3000 Particle Filter'])
plt.savefig('number_of_particle2', bbox_inches='tight')

# Plot y-position
plt.figure('y position vs time step')
plt.plot(t, motion_states[1:, 1])
plt.scatter(t, measurement_states[1:, 1])
plt.plot(t, m0[:, 1])
plt.plot(t, m1[:, 1])
plt.plot(t, m2[:, 1])
plt.plot(t, m3[:, 1])
plt.plot(t, m4[:, 1])
plt.plot(t, m5[:, 1])
plt.xlabel('Time step')
plt.ylabel('y position')
plt.title('y position vs time step')
plt.legend(['Position', 'Measured Postion', 'N=100 Particle Filter', 'N=300 Particle Filter', 'N=500 Particle Filter', 'N=1000 Particle Filter', 'N=1500 Particle Filter', 'N=3000 Particle Filter'])
plt.savefig('number_of_particle3', bbox_inches='tight')

# Plot aver_mse
plt.figure('average MSE vs number of particles')
plt.plot([n0, n1, n2, n3, n4, n5], [a_mse0, a_mse1, a_mse2, a_mse3, a_mse4, a_mse5], linewidth=1)
plt.xlabel('Number of particles')
plt.ylabel('Average MSE')
plt.title('Average MSE vs number of particles')
plt.savefig('number_of_particle4.pdf', bbox_inches='tight')

print([a_mse0, a_mse1, a_mse2, a_mse3, a_mse4, a_mse5])
