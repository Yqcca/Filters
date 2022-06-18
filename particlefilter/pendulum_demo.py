from math import sin
import numpy as np
import matplotlib.pyplot as plt
from particlefilter import nonlinear_gaussian_adaptive_resampling_particle_filter, nonlinear_gaussian_bootstrap_filter,\
    nonlinear_gaussian_resampling_particle_filter, nonlinear_gaussian_sampling_particle_filter
from EKF import predict, update

dt = 0.1
q = q_c = 1/2
g = 9.81
sigma = 1/4

Q = np.array([[(q*dt**3)/3, (q*dt**2)/2],
             [(q*dt**2)/2, q*dt]])

R = np.array([[sigma**2]])


def f11(x):
    y = np.zeros(x.shape)
    y[0] = x[0] + x[1]*dt
    y[1] = x[1] - g*sin(x[0])*dt
    return y


def h11(x):
    y = np.zeros(1)
    y = sin(x[0])
    return y

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
num_steps = 100
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

m_0 = np.array([[1.5, 0]]).T
P_0 = Q
m_current = m_0.copy()
P_current = P_0.copy()

for i in range(num_steps):
    predicted_m, predicted_P = predict(f, F, Q, m_current, P_current)
    y = measurement_angle[i+1]
    m_current, P_current = update(h, H, R, y, predicted_m, predicted_P)
    filtered_states.append(m_current)

filtered_states = np.array(filtered_states)


N = 1500
w0, m0 = nonlinear_gaussian_resampling_particle_filter(f11, Q, h11, R, N, num_steps, measurement_angle)
w1, m1 = nonlinear_gaussian_adaptive_resampling_particle_filter(f11, Q, h11, R, N, num_steps, measurement_angle)
w2, m2 = nonlinear_gaussian_bootstrap_filter(f11, Q, h11, R, N, num_steps, measurement_angle)
w3, m3 = nonlinear_gaussian_sampling_particle_filter(f11, Q, h11, R, N, num_steps, measurement_angle)

m00 = [sin(i) for i in m0[:, 0]]
m11 = [sin(i) for i in m1[:, 0]]
m22 = [sin(i) for i in m2[:, 0]]
m33 = [sin(i) for i in m3[:, 0]]
m44 = np.sin(filtered_states[:, 0])

# Plot
t = [i for i in range(num_steps+1)]

hy = [sin(i) for i in true_states[:, 0]]
plt.figure('angle position vs time step')
plt.plot(t, hy)
plt.scatter(t, measurement_angle)
plt.plot(t[1:], m00)
plt.plot(t[1:], m11)
plt.plot(t[1:], m22)
plt.plot(t[1:], m33)
plt.plot(t[1:], m44)
plt.xlabel('Time step')
plt.ylabel('Angle position')
plt.legend(['Angle Position', 'Measured Angle Postion', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF', 'Extended Kalman Filter'])
plt.title('Angle position vs time step')
plt.savefig('pendulum1.pdf', bbox_inches='tight')

benchmark = np.sin(true_states[:, 0])
# Compute avg MSE in position
def avg_mse(m, motion_states, num_steps):
    mse_x = 0
    for i in range(num_steps):
        mse_x += (m[i] - motion_states[i])**2
    return mse_x/num_steps

a_mse0= avg_mse(m00, benchmark, num_steps)
a_mse1= avg_mse(m11, benchmark, num_steps)
a_mse2= avg_mse(m22, benchmark, num_steps)
a_mse3= avg_mse(m33, benchmark, num_steps)
a_mse4= avg_mse(m44, benchmark, num_steps)

print([a_mse0, a_mse1, a_mse2, a_mse3, a_mse4])

# Plot the MSE at each step
def step_mse(m, motion_states, num_steps):
    mse_x = []
    for i in range(num_steps):
        mse_x.append((m[i] - motion_states[i])**2)
    return mse_x

mse_0 = step_mse(m00, benchmark, num_steps)
mse_1 = step_mse(m11, benchmark, num_steps)
mse_2 = step_mse(m22, benchmark, num_steps)
mse_3 = step_mse(m33, benchmark, num_steps)
mse_4 = step_mse(m44, benchmark, num_steps)

plt.figure('MSE in angle position vs time step')
plt.plot(t[1:], mse_0, linewidth = 1)
plt.plot(t[1:], mse_1, linewidth = 1)
plt.plot(t[1:], mse_2, linewidth = 1)
plt.plot(t[1:], mse_3, linewidth = 1)
plt.plot(t[1:], mse_4, linewidth = 1)
plt.xlabel('Time step')
plt.ylabel('MSE in angle position')
plt.title('MSE in angle position vs time step')
plt.legend(['SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF', 'Extended Kalman Filter'])
plt.savefig('pendulum2.pdf', bbox_inches='tight')
