import numpy as np
import math
import matplotlib.pyplot as plt
from particlefilter import linear_gaussian_adaptive_resampling_particle_filter, linear_gaussian_bootstrap_filter,\
    linear_gaussian_resampling_particle_filter, linear_gaussian_sampling_particle_filter
from KF import predict, update
import seaborn as sns

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
N = 1000

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
t = [i for i in range(num_steps)]

# Particle Filter
x_record0, w0, m0, var0 = linear_gaussian_resampling_particle_filter(A, Q, H, R, N, num_steps, measurement_states)
x_record1, w1, m1, var1 = linear_gaussian_adaptive_resampling_particle_filter(A, Q, H, R, N, num_steps, measurement_states)
x_record2, w2, m2, var2 = linear_gaussian_bootstrap_filter(A, Q, H, R, N, num_steps, measurement_states)
x_record3, w3, m3, var3 = linear_gaussian_sampling_particle_filter(A, Q, H, R, N, num_steps, measurement_states)

plt.figure('x position pdf1')
plt.axvline(x=motion_states[25][0], ls=':')
sns.histplot(x = x_record0[25][:, 0], weights = w0[25], color = 'purple', kde=True)
sns.histplot(x = x_record1[25][:, 0], weights = w1[25], color = 'blue', kde=True)
sns.histplot(x = x_record2[25][:, 0], weights = w2[25], color = 'green', kde=True)
sns.histplot(x = x_record3[25][:, 0], weights = w3[25], color = 'red', kde=True)
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf in x-direction at time step 25')
plt.legend(['True x-position at step 25', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking8.pdf', bbox_inches='tight')

plt.figure('y position pdf1')
plt.axvline(x=motion_states[25][1], ls=':')
sns.histplot(x = x_record0[25][:, 1], weights = w0[25], color = 'purple', kde=True)
sns.histplot(x = x_record1[25][:, 1], weights = w1[25], color = 'blue', kde=True)
sns.histplot(x = x_record2[25][:, 1], weights = w2[25], color = 'green', kde=True)
sns.histplot(x = x_record3[25][:, 1], weights = w3[25], color = 'red', kde=True)
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf in y-direction at time step 25')
plt.legend(['True y-position at step 25', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking9.pdf', bbox_inches='tight')


plt.figure('x position pdf2')
plt.axvline(x=motion_states[50][0], ls=':')
sns.histplot(x = x_record0[50][:, 0], weights = w0[50], color = 'purple', kde=True)
sns.histplot(x = x_record1[50][:, 0], weights = w1[50], color = 'blue', kde=True)
sns.histplot(x = x_record2[50][:, 0], weights = w2[50], color = 'green', kde=True)
sns.histplot(x = x_record3[50][:, 0], weights = w3[50], color = 'red', kde=True)
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf in x-direction at time step 50')
plt.legend(['True x-position at step 50', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking10.pdf', bbox_inches='tight')

plt.figure('y position pdf2')
plt.axvline(x=motion_states[50][1], ls=':')
sns.histplot(x = x_record0[50][:, 1], weights = w0[50], color = 'purple',kde=True)
sns.histplot(x = x_record1[50][:, 1], weights = w1[50], color = 'blue', kde=True)
sns.histplot(x = x_record2[50][:, 1], weights = w2[50], color = 'green',kde=True)
sns.histplot(x = x_record3[50][:, 1], weights = w3[50], color = 'red', kde=True)
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf in y-direction at time step 50')
plt.legend(['True y-position at step 50', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking11.pdf', bbox_inches='tight')

plt.figure('x position pdf3')
plt.axvline(x=motion_states[75][0], ls=':')
sns.histplot(x = x_record0[75][:, 0], weights = w0[75], color = 'purple',kde=True)
sns.histplot(x = x_record1[75][:, 0], weights = w1[75], color = 'blue', kde=True)
sns.histplot(x = x_record2[75][:, 0], weights = w2[75], color = 'green',kde=True)
sns.histplot(x = x_record3[75][:, 0], weights = w3[75], color = 'red', kde=True)
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf in x-direction at time step 75')
plt.legend(['True x-position at time step 75', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking12.pdf', bbox_inches='tight')

plt.figure('y position pdf3')
plt.axvline(x=motion_states[75][1], ls=':')
sns.histplot(x = x_record0[75][:, 1], weights = w0[75], color = 'purple',kde=True)
sns.histplot(x = x_record1[75][:, 1], weights = w1[75], color = 'blue', kde=True)
sns.histplot(x = x_record2[75][:, 1], weights = w2[75], color = 'green', kde=True)
sns.histplot(x = x_record3[75][:, 1], weights = w3[75], color = 'red', kde=True)
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf in y-direction at time step 75')
plt.legend(['True y-position at time step 75', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking13.pdf', bbox_inches='tight')


var0_x = var0[:, 0]
var0_y = var0[:, 1]
var1_x = var1[:, 0]
var1_y = var1[:, 1]
var2_x = var2[:, 0]
var2_y = var2[:, 1]
var3_x = var3[:, 0]
var3_y = var3[:, 1]

plt.figure('x variance')
plt.plot(t, var0_x)
plt.plot(t, var1_x)
plt.plot(t, var2_x)
plt.plot(t, var3_x)
plt.xlabel('Time step')
plt.ylabel('x variance')
plt.title('x variance vs x time step')
plt.legend(['SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking6.pdf', bbox_inches='tight')

plt.figure('y variance')
plt.plot(t, var0_y)
plt.plot(t, var1_y)
plt.plot(t, var2_y)
plt.plot(t, var3_y)
plt.xlabel('Time step')
plt.ylabel('y variance')
plt.title('y variance vs x time step')
plt.legend(['SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('car tracking7.pdf', bbox_inches='tight')


# KF
m_0 = np.array([0, 0, 0, 0])
P_0 = np.zeros((4,4))
filtered_states = []
m_current = m_0.copy()
P_current = P_0.copy()

for i in range(num_steps):
    predicted_m, predicted_P = predict(A, Q, m_current, P_current)
    y = measurement_states[i+1]
    m_current, P_current = update(H, R, y, predicted_m, predicted_P)
    filtered_states.append(m_current)
m4 = np.array(filtered_states)

# Plot the x, y pos of the states
plt.figure('Car Position')
plt.plot(motion_states[:, 0], motion_states[:, 1])
plt.scatter(measurement_states[:, 0], measurement_states[:, 1])
plt.plot(m0[:, 0], m0[:, 1])
plt.plot(m1[:, 0], m1[:, 1])
plt.plot(m2[:, 0], m2[:, 1])
plt.plot(m3[:, 0], m3[:, 1])
plt.plot(m4[:, 0], m4[:, 1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('y position vs x position')
plt.legend(['Position', 'Measured Postion', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF', 'Kalman Filter'])
plt.savefig('car tracking1.pdf', bbox_inches='tight')

# Plot x position
plt.figure('x position vs time step')
plt.plot(t, motion_states[1:, 0])
plt.scatter(t, measurement_states[1:, 0])
plt.plot(t, m0[:, 0])
plt.plot(t, m1[:, 0])
plt.plot(t, m2[:, 0])
plt.plot(t, m3[:, 0])
plt.plot(t, m4[:, 0])
plt.xlabel('Time step')
plt.ylabel('x position')
plt.title('x position vs time step')
plt.legend(['Position', 'Measured Postion', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF', 'Kalman Filter'])
plt.savefig('car tracking2.pdf', bbox_inches='tight')

# Plot x position
plt.figure('y position vs time step')
plt.plot(t, motion_states[1:, 1])
plt.scatter(t, measurement_states[1:, 1])
plt.plot(t, m0[:, 1])
plt.plot(t, m1[:, 1])
plt.plot(t, m2[:, 1])
plt.plot(t, m3[:, 1])
plt.plot(t, m4[:, 1])
plt.xlabel('Time step')
plt.ylabel('y position')
plt.title('y position vs time step')
plt.legend(['Position', 'Measured Postion', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF', 'Kalman Filter'])
plt.savefig('car tracking3.pdf', bbox_inches='tight')

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

print([a_mse0, a_mse1, a_mse2, a_mse3, a_mse4])
