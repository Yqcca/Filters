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
x_record0, w0, m0, var0 = nonlinear_gaussian_resampling_particle_filter(f11, Q, h11, R, N, num_steps, measurement_angle)
x_record1, w1, m1, var1 = nonlinear_gaussian_adaptive_resampling_particle_filter(f11, Q, h11, R, N, num_steps, measurement_angle)
x_record2, w2, m2, var2 = nonlinear_gaussian_bootstrap_filter(f11, Q, h11, R, N, num_steps, measurement_angle)
x_record3, w3, m3, var3 = nonlinear_gaussian_sampling_particle_filter(f11, Q, h11, R, N, num_steps, measurement_angle)

plt.figure('x position pdf1')
plt.axvline(x=sin(true_states[25][0]), ls=':')
plt.hist([sin(i) for i in x_record0[25][:,0]], color = 'purple', weights = w0[25])
plt.hist([sin(i) for i in x_record1[25][:, 0]], color = 'blue', weights = w1[25])
plt.hist([sin(i) for i in x_record2[25][:, 0]], color = 'green', weights = w2[25])
plt.hist([sin(i) for i in x_record3[25][:, 0]], color = 'red', weights = w3[25])
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf of angle position at time step 25')
plt.legend(['True x-position at step 25', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum4.pdf', bbox_inches='tight')

plt.figure('x position pdf2')
plt.axvline(x=sin(true_states[26][0]), ls=':')
plt.hist([sin(i) for i in x_record0[26][:,0]], color = 'purple', weights = w0[26])
plt.hist([sin(i) for i in x_record1[26][:, 0]], color = 'blue', weights = w1[26])
plt.hist([sin(i) for i in x_record2[26][:, 0]], color = 'green', weights = w2[26])
plt.hist([sin(i) for i in x_record3[26][:, 0]], color = 'red', weights = w3[26])
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf of angle position at time step 25')
plt.legend(['True x-position at step 25', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum7.pdf', bbox_inches='tight')

plt.figure('x position pdf3')
plt.axvline(x=sin(true_states[50][0]), ls=':')
plt.hist([sin(i) for i in x_record0[50][:, 0]], color = 'purple', weights = w0[50])
plt.hist([sin(i) for i in x_record1[50][:, 0]], color = 'blue', weights = w1[50])
plt.hist([sin(i) for i in x_record2[50][:, 0]], color = 'green', weights = w2[50])
plt.hist([sin(i) for i in x_record3[50][:, 0]], color = 'red', weights = w3[50])
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf of angle position at time step 50')
plt.legend(['True angle position at step 50', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum5.pdf', bbox_inches='tight')

plt.figure('x position pdf4')
plt.axvline(x=sin(true_states[50][0]), ls=':')
plt.hist([sin(i) for i in x_record0[51][:, 0]], color = 'purple', weights = w0[51])
plt.hist([sin(i) for i in x_record1[51][:, 0]], color = 'blue', weights = w1[51])
plt.hist([sin(i) for i in x_record2[51][:, 0]], color = 'green', weights = w2[51])
plt.hist([sin(i) for i in x_record3[51][:, 0]], color = 'red', weights = w3[51])
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf of angle position at time step 50')
plt.legend(['True angle position at step 50', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum8.pdf', bbox_inches='tight')

plt.figure('x position pdf5')
plt.axvline(x=sin(true_states[75][0]), ls=':')
plt.hist([sin(i) for i in x_record0[75][:, 0]], color = 'purple', weights = w0[75])
plt.hist([sin(i) for i in x_record1[75][:, 0]], color = 'blue', weights = w1[75])
plt.hist([sin(i) for i in x_record2[75][:, 0]], color = 'green', weights = w2[75])
plt.hist([sin(i) for i in x_record3[75][:, 0]], color = 'red', weights = w3[75])
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf of angle position at time step 75')
plt.legend(['True angle position at step 75', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum6.pdf', bbox_inches='tight')

plt.figure('x position pdf6')
plt.axvline(x=sin(true_states[76][0]), ls=':')
plt.hist([sin(i) for i in x_record0[76][:, 0]], color = 'purple', weights = w0[76])
plt.hist([sin(i) for i in x_record1[76][:, 0]], color = 'blue', weights = w1[76])
plt.hist([sin(i) for i in x_record2[76][:, 0]], color = 'green', weights = w2[76])
plt.hist([sin(i) for i in x_record3[76][:, 0]], color = 'red', weights = w3[76])
plt.xlabel('Samples')
plt.ylabel('Weights')
plt.title('Particle pdf of angle position at time step 75')
plt.legend(['True angle position at step 75', 'SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum9.pdf', bbox_inches='tight')


m00 = [sin(i) for i in m0[:, 0]]
m11 = [sin(i) for i in m1[:, 0]]
m22 = [sin(i) for i in m2[:, 0]]
m33 = [sin(i) for i in m3[:, 0]]
m44 = np.sin(filtered_states[:, 0])

# Plot
t1 = [i for i in range(num_steps)]
var0 = var0[:, 0]
var1 = var1[:, 0]
var2 = var2[:, 0]
var3 = var3[:, 0]

plt.figure('Angle position variance')
plt.plot(t1, var0)
plt.plot(t1, var1)
plt.plot(t1, var2)
plt.plot(t1, var3)
plt.xlabel('Time step')
plt.ylabel('Angle position variance')
plt.title('Angle position variance vs x time step')
plt.legend(['SIR PF', 'Adaptive SIR PF', 'Bootstrap Filter', 'SIS PF'])
plt.savefig('pendulum3.pdf', bbox_inches='tight')

hy = [sin(i) for i in true_states[:, 0]]
plt.figure('angle position vs time step')
plt.plot(t1, hy[1:])
plt.scatter(t1, measurement_angle[1:])
plt.plot(t1, m00)
plt.plot(t1, m11)
plt.plot(t1, m22)
plt.plot(t1, m33)
plt.plot(t1, m44)
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
