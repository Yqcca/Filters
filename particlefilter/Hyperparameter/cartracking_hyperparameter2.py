import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Linear Gaussian case
def prior_sample(U, Q, N):
    x_0 = np.random.multivariate_normal(mean=U, cov=Q, size=N)
    w_0 = np.array([1/N for i in range(N)])
    return x_0, w_0


def normalized_weight(w):
    v = []
    s = np.sum(w)
    for i in range(len(w)):
        v.append(w[i]/s)
    return np.array(v)


def resampling(w, x):
    ind = [i for i in range(len(w))]
    n = np.zeros(x.shape)
    for i in range(len(w)):
        n[i] = x[np.random.choice(ind, size=1, p=w)]
    m = np.zeros(len(w))
    for i in range(len(w)):
        m[i] = 1/len(w)
    return m, n


def check_resampling(w, x):
    rneff = 0
    for i in range(len(w)):
        rneff += w[i]**2
    neff = 1/rneff
    if neff < len(w)/10:
        return resampling(w, x)
    else:
        return w, x


def linear_gaussian_importance_distribution1(prev_x, y, A, Q):
    while prev_x.shape != y.shape:
        y = np.append(y, 0)
    m = 0.5*A@prev_x + 0.5*y
    f = stats.multivariate_normal(mean=m, cov=Q)
    return f
def linear_gaussian_importance_distribution2(prev_x, y, A, Q):
    while prev_x.shape != y.shape:
        y = np.append(y, 0)
    m = 0.4*A@prev_x + 0.6*y
    f = stats.multivariate_normal(mean=m, cov=Q)
    return f
def linear_gaussian_importance_distribution3(prev_x, y, A, Q):
    while prev_x.shape != y.shape:
        y = np.append(y, 0)
    m = 0.3*A@prev_x + 0.7*y
    f = stats.multivariate_normal(mean=m, cov=Q)
    return f
def linear_gaussian_importance_distribution4(prev_x, y, A, Q):
    while prev_x.shape != y.shape:
        y = np.append(y, 0)
    m = 0.2*A@prev_x + 0.8*y
    f = stats.multivariate_normal(mean=m, cov=Q)
    return f

def linear_gaussian_adaptive_resampling_particle_filter1(A, Q, H, R, N, T, y):
    # Initialization
    n = A.shape[0]
    prev_x, prev_w = prior_sample(np.zeros(n), Q, N)
    w_record = [prev_w]
    var = np.zeros((T, n))
    m_final = np.zeros((T, n))
    for i in range(N):
        m_final[0] += prev_w[i] * prev_x[i]

    # Calculate weight for each time step
    for k in range(1, T):
        x = np.zeros((N, n))
        w = np.zeros(N)
        for i in range(N):
            f = linear_gaussian_importance_distribution1(prev_x[i], y[k], A, Q)
            x[i] = np.array(f.rvs(size=1))
            g1 = stats.multivariate_normal(mean=H@x[i], cov=R)
            g2 = stats.multivariate_normal(mean=A@prev_x[i], cov=Q)
            w[i] = (prev_w[i]*g1.pdf(y[k])*g2.pdf(x[i])/f.pdf(x[i])) + np.finfo(float).eps
        w = normalized_weight(w)
        w, x = check_resampling(w, x)
        m_final[k] = np.average(x, weights=w, axis=0)
        var[k]  = np.average((x - m_final[k])**2, weights=w, axis=0)
        prev_x = x
        prev_w = w
        w_record.append(prev_w)
    return w_record, m_final, var

def linear_gaussian_adaptive_resampling_particle_filter2(A, Q, H, R, N, T, y):
    # Initialization
    n = A.shape[0]
    prev_x, prev_w = prior_sample(np.zeros(n), Q, N)
    w_record = [prev_w]
    m_final = np.zeros((T, n))
    for i in range(N):
        m_final[0] += prev_w[i] * prev_x[i]

    # Calculate weight for each time step
    for k in range(1, T):
        x = np.zeros((N, n))
        w = np.zeros(N)
        var = np.zeros((T, n))
        for i in range(N):
            f = linear_gaussian_importance_distribution2(prev_x[i], y[k], A, Q)
            x[i] = np.array(f.rvs(size=1))
            g1 = stats.multivariate_normal(mean=H@x[i], cov=R)
            g2 = stats.multivariate_normal(mean=A@prev_x[i], cov=Q)
            w[i] = (prev_w[i]*g1.pdf(y[k])*g2.pdf(x[i])/f.pdf(x[i])) + np.finfo(float).eps
        w = normalized_weight(w)
        w, x = check_resampling(w, x)
        m_final[k] = np.average(x, weights=w, axis=0)
        var[k]  = np.average((x - m_final[k])**2, weights=w, axis=0)
        prev_x = x
        prev_w = w
        w_record.append(prev_w)
    return w_record, m_final, var

def linear_gaussian_adaptive_resampling_particle_filter3(A, Q, H, R, N, T, y):
    # Initialization
    n = A.shape[0]
    prev_x, prev_w = prior_sample(np.zeros(n), Q, N)
    w_record = [prev_w]
    m_final = np.zeros((T, n))
    for i in range(N):
        m_final[0] += prev_w[i] * prev_x[i]

    # Calculate weight for each time step
    for k in range(1, T):
        x = np.zeros((N, n))
        w = np.zeros(N)
        var = np.zeros((T, n))
        for i in range(N):
            f = linear_gaussian_importance_distribution3(prev_x[i], y[k], A, Q)
            x[i] = np.array(f.rvs(size=1))
            g1 = stats.multivariate_normal(mean=H@x[i], cov=R)
            g2 = stats.multivariate_normal(mean=A@prev_x[i], cov=Q)
            w[i] = (prev_w[i]*g1.pdf(y[k])*g2.pdf(x[i])/f.pdf(x[i])) + np.finfo(float).eps
        w = normalized_weight(w)
        w, x = check_resampling(w, x)
        m_final[k] = np.average(x, weights=w, axis=0)
        var[k]  = np.average((x - m_final[k])**2, weights=w, axis=0)
        prev_x = x
        prev_w = w
        w_record.append(prev_w)
    return w_record, m_final, var

def linear_gaussian_adaptive_resampling_particle_filter4(A, Q, H, R, N, T, y):
    # Initialization
    n = A.shape[0]
    prev_x, prev_w = prior_sample(np.zeros(n), Q, N)
    w_record = [prev_w]
    m_final = np.zeros((T, n))
    for i in range(N):
        m_final[0] += prev_w[i] * prev_x[i]

    # Calculate weight for each time step
    for k in range(1, T):
        x = np.zeros((N, n))
        w = np.zeros(N)
        var = np.zeros((T, n))
        for i in range(N):
            f = linear_gaussian_importance_distribution4(prev_x[i], y[k], A, Q)
            x[i] = np.array(f.rvs(size=1))
            g1 = stats.multivariate_normal(mean=H@x[i], cov=R)
            g2 = stats.multivariate_normal(mean=A@prev_x[i], cov=Q)
            w[i] = (prev_w[i]*g1.pdf(y[k])*g2.pdf(x[i])/f.pdf(x[i])) + np.finfo(float).eps
        w = normalized_weight(w)
        w, x = check_resampling(w, x)
        m_final[k] = np.average(x, weights=w, axis=0)
        var[k]  = np.average((x - m_final[k])**2, weights=w, axis=0)
        prev_x = x
        prev_w = w
        w_record.append(prev_w)
    return w_record, m_final, var

def linear_gaussian_bootstrap_filter(A, Q, H, R, N, T, y):
    # Initialization
    n = A.shape[0]
    w_record = []
    var = np.zeros((T, n))
    prev_x = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size=N)
    m_final = np.zeros((T, n))

    # Calculate weight for each time step
    for k in range(T):
        x = np.zeros((N, n))
        w = np.zeros(N)
        for i in range(N):
            g1 = stats.multivariate_normal(mean=A@prev_x[i], cov=Q)
            x[i] = np.array(g1.rvs(size=1))
            g2 = stats.multivariate_normal(mean=H@x[i], cov=R)
            w[i] = g2.pdf(y[k])
        w = normalized_weight(w)
        w, x = resampling(w, x)
        m_final[k] = np.average(x, weights=w, axis=0)
        var[k]  = np.average((x - m_final[k])**2, weights=w, axis=0)
        prev_x = x
        w_record.append(w)
    return w_record, m_final, var

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

num_steps = 100
x_0, y_0, vx_0, vy_0 = 0, 0, 0, 0
N = 1000

motion_states = [np.array([x_0, y_0, vx_0, vy_0])]
for i in range(num_steps):
    motion_noise = np.random.multivariate_normal(mean=np.array([0, 0, 0, 0]), cov=Q)
    new_state = A @ motion_states[-1] + motion_noise
    motion_states.append(new_state)
motion_states = np.array(motion_states)

measurement_states = [np.array([x_0, y_0])]
for i in range(num_steps):
    measurement_noise = np.random.multivariate_normal(mean=np.array([0, 0]), cov=R)
    new_measurement = H @ motion_states[i] + measurement_noise
    measurement_states.append(new_measurement)
measurement_states = np.array(measurement_states)

w0, m0, var0 = linear_gaussian_adaptive_resampling_particle_filter1(A, Q, H, R, N, num_steps, measurement_states)
w1, m1, var1 = linear_gaussian_adaptive_resampling_particle_filter2(A, Q, H, R, N, num_steps, measurement_states)
w2, m2, var2 = linear_gaussian_adaptive_resampling_particle_filter3(A, Q, H, R, N, num_steps, measurement_states)
w3, m3, var3 = linear_gaussian_adaptive_resampling_particle_filter4(A, Q, H, R, N, num_steps, measurement_states)
w4, m4, var4 = linear_gaussian_bootstrap_filter(A, Q, H, R, N, num_steps, measurement_states)

var0_x = var0[:, 0]
var0_y = var0[:, 1]
var1_x = var1[:, 0]
var1_y = var1[:, 1]
var2_x = var2[:, 0]
var2_y = var2[:, 1]
var3_x = var3[:, 0]
var3_y = var3[:, 1]
var4_x = var4[:, 0]
var4_y = var4[:, 1]

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

# Plot the x, y pos of the states

t = [i for i in range(num_steps)]
plt.figure('x position vs time step')
plt.plot(t, motion_states[1:, 0], linewidth = 1)
plt.scatter(t, measurement_states[1:, 0])
plt.plot(t, m0[:, 0], linewidth = 1)
plt.plot(t, m1[:, 0], linewidth = 1)
plt.plot(t, m2[:, 0], linewidth = 1)
plt.plot(t, m3[:, 0], linewidth = 1)
plt.plot(t, m4[:, 0], linewidth = 1)
plt.xlabel('Time step')
plt.ylabel('x position')
plt.title('x position vs time step')
plt.legend(['Position', 'Measured Postion', 'Type1 SIR PF', 'Type2 SIR PF', 'Type3 SIR PF', 'Type4 SIR PF', 'Bootstrap Filter'])
plt.savefig('model1', bbox_inches='tight')

plt.figure('xyposition vs time step')
plt.plot(t, motion_states[1:, 1], linewidth = 1)
plt.scatter(t, measurement_states[1:, 1])
plt.plot(t, m0[:, 1], linewidth = 1)
plt.plot(t, m1[:, 1], linewidth = 1)
plt.plot(t, m2[:, 1], linewidth = 1)
plt.plot(t, m3[:, 1], linewidth = 1)
plt.plot(t, m4[:, 1], linewidth = 1)
plt.xlabel('Time step')
plt.ylabel('y position')
plt.title('y position vs time step')
plt.legend(['Position', 'Measured Postion', 'Type1 SIR PF', 'Type2 SIR PF', 'Type3 SIR PF', 'Type4 SIR PF', 'Bootstrap Filter'])
plt.savefig('model2', bbox_inches='tight')

print([a_mse0, a_mse1, a_mse2, a_mse3, a_mse4])

plt.figure('x variance')
plt.plot(t, var0_x)
plt.plot(t, var1_x)
plt.plot(t, var2_x)
plt.plot(t, var3_x)
plt.plot(t, var4_x)
plt.xlabel('Time step')
plt.ylabel('x variance')
plt.title('x variance vs x time step')
plt.legend(['Type1 SIR PF', 'Type2 SIR PF', 'Type3 SIR PF', 'Type4 SIR PF', 'Bootstrap Filter'])
plt.savefig('model3', bbox_inches='tight')

plt.figure('y variance')
plt.plot(t, var0_y)
plt.plot(t, var1_y)
plt.plot(t, var2_y)
plt.plot(t, var3_y)
plt.plot(t, var4_y)
plt.xlabel('Time step')
plt.ylabel('y variance')
plt.title('y variance vs x time step')
plt.legend(['Type1 SIR PF', 'Type2 SIR PF', 'Type3 SIR PF', 'Type4 SIR PF', 'Bootstrap Filter'])
plt.savefig('model4', bbox_inches='tight')
