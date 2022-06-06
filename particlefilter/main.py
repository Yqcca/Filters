import numpy as np
import matplotlib.pyplot as plt
from particlefilter import linear_gaussian_particle_filter

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

#Simulate Motion
num_steps = 50
x_0, y_0, vx_0, vy_0 = 0, 0, 0, 0
motion_states = [np.array([x_0, y_0, vx_0, vy_0])]
for i in range(num_steps):
    motion_noise = np.random.multivariate_normal(mean=np.array([0,0,0,0]), cov=Q)
    new_state = A @ motion_states[-1] + motion_noise
    motion_states.append(new_state)
motion_states = np.array(motion_states)

#Simulate Measurement
measurement_states = [np.array([x_0, y_0])]
for i in range(num_steps):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R)
    new_measurement = H @ motion_states[i] + measurement_noise
    measurement_states.append(new_measurement)
measurement_states = np.array(measurement_states)

m = linear_gaussian_particle_filter(A, Q, H, R, 200, num_steps, measurement_states)
print(m)


#Plot the x, y pos of the states
plt.plot(motion_states[:,0], motion_states[:,1])
plt.scatter(measurement_states[:,0], measurement_states[:,1])
plt.scatter(m[:,0], m[:,1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['Position', 'Measured Postion', 'Particle Filter Postition'])
plt.show()
