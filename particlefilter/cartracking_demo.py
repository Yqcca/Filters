import numpy as np
import math
import matplotlib.pyplot as plt
from particlefilter import linear_gaussian_adaptive_resampling_particle_filter, linear_gaussian_bootstrap_filter,\
    linear_gaussian_resampling_particle_filter

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
num_steps = 60
x_0, y_0, vx_0, vy_0 = 0, 0, 0, 0
N = 100

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

w1, m1 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, N, num_steps, measurement_states)
w2, m2 = linear_gaussian_bootstrap_filter(A, Q, H, R, N, num_steps, measurement_states)

# Plot the x, y pos of the states
plt.figure('Car Position')
plt.plot(motion_states[:, 0], motion_states[:, 1])
plt.scatter(measurement_states[:, 0], measurement_states[:, 1])
plt.plot(m1[:, 0], m1[:, 1])
plt.plot(m2[:, 0], m2[:, 1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('y position vs x position')
plt.legend(['Position', 'Measured Postion', 'Particle Filter', 'Bootstrap_Filter'])
plt.savefig('tmp1.pdf', bbox_inches='tight')

t = [i for i in range(num_steps)]

# Draw vertical gaussian
def draw_gaussian_at(mu, sd, height, xpos):
    support = np.linspace(mu-3*sd, mu+3*sd)
    gaussian = np.exp((-0.5*((support-mu)/sd)**2))/(sd*math.sqrt(2*math.pi))
    gaussian /= gaussian.max()
    gaussian *= height
    return plt.plot(gaussian + xpos, support),\
         plt.hlines(y=mu, xmin=gaussian.min()+xpos, xmax=gaussian.max()+xpos, linewidth=2, color='r', linestyle='--')

# Compute covariance:
def covar(w, A, N, a):
    l_cov = np.zeros(A.shape)
    for i in range(N):
        l_cov += w[a][i] * A
    return l_cov[0, 0], l_cov[1, 1]

# First covariance 
l_cov1_x, l_cov1_y = covar(w1, Q, N, 1)
l_cov2_x, l_cov2_y = covar(w2, R, N, 1)

# Last covariance
l_cov3_x, l_cov3_y =  covar(w1, Q, N, -1)
l_cov4_x, l_cov4_y = covar(w2, R, N, -1)

# Middle covariance
mid = num_steps // 2
l_cov5_x, l_cov5_y = covar(w1, Q, N, mid)
l_cov6_x, l_cov6_y = covar(w2, R, N, mid)

# Plot x position
plt.figure('x position vs time step')
plt.plot(t, motion_states[1:, 0])
plt.scatter(t, measurement_states[1:, 0])
plt.plot(t, m1[:, 0])
plt.plot(t, m2[:, 0])
draw_gaussian_at(mu=m1[1, 0], sd=math.sqrt(l_cov1_x), height=5, xpos=1)
draw_gaussian_at(mu=m2[1, 0], sd=math.sqrt(l_cov2_x), height=1, xpos=1)

draw_gaussian_at(mu=m1[-1, 0], sd=math.sqrt(l_cov3_x), height=5, xpos=num_steps)
draw_gaussian_at(mu=m2[-1, 0], sd=math.sqrt(l_cov4_x), height=1, xpos=num_steps)

draw_gaussian_at(mu=m1[mid, 0], sd=math.sqrt(l_cov5_x), height=5, xpos=mid)
draw_gaussian_at(mu=m2[mid, 0], sd=math.sqrt(l_cov6_x), height=1, xpos=mid)

plt.xlabel('time step')
plt.ylabel('x position')
plt.title('x position vs time step')
plt.legend(['Position', 'Measured Postion', 'Particle Filter', 'Bootstrap_Filter'])
plt.savefig('tmp2.pdf', bbox_inches='tight')

# Plot x position
plt.figure('y position vs time step')
plt.plot(t, motion_states[1:, 1])
plt.scatter(t, measurement_states[1:, 1])
plt.plot(t, m1[:, 1])
plt.plot(t, m2[:, 1])
draw_gaussian_at(mu=m1[1, 1], sd=math.sqrt(l_cov1_y), height=5, xpos=1)
draw_gaussian_at(mu=m2[1, 1], sd=math.sqrt(l_cov2_y), height=1, xpos=1)

draw_gaussian_at(mu=m1[-1, 1], sd=math.sqrt(l_cov3_y), height=5, xpos=num_steps)
draw_gaussian_at(mu=m2[-1, 1], sd=math.sqrt(l_cov4_y), height=1, xpos=num_steps)

draw_gaussian_at(mu=m1[mid, 1], sd=math.sqrt(l_cov5_y), height=5, xpos=mid)
draw_gaussian_at(mu=m2[mid, 1], sd=math.sqrt(l_cov6_y), height=1, xpos=mid)

plt.xlabel('time step')
plt.ylabel('y position')
plt.title('y position vs time step')
plt.legend(['Position', 'Measured Postion', 'Particle Filter', 'Bootstrap_Filter'])

plt.savefig('tmp3.pdf', bbox_inches='tight')

'''# Mse of adaptive resampling particle filter
# Compute average MSE
def aver_mse(m, motion_states, num_steps):
    mse_x = 0
    mse_y = 0
    for i in range(num_steps):
        mse_x += (m[i, 0] - motion_states[i, 0])**2
        mse_y += (m[i, 1] - motion_states[i, 1])**2
    mse_x = mse_x / num_steps
    mse_y = mse_y / num_steps
    return mse_x, mse_y

mse_x1 = 0
mse_y1 = 0
for i in range(num_steps):
    mse_x1 += (m1[i, 0] - motion_states[i, 0])**2
    mse_y1 += (m1[i, 1] - motion_states[i, 1])**2
mse_x1 = mse_x1 / num_steps
mse_y1 = mse_y1 / num_steps

# Mse of bootstrap filter
mse_x2 = 0
mse_y2 = 0
for i in range(num_steps):
    mse_x2 += (m2[i, 0] - motion_states[i, 0])**2
    mse_y2 += (m2[i, 1] - motion_states[i, 1])**2
mse_x2 = mse_x2 / num_steps
mse_y2 = mse_y2 / num_steps

print([mse_x1, mse_y1], [mse_x2, mse_y2])'''
