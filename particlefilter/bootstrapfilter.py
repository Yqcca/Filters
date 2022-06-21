import numpy as np
from scipy import stats

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

def linear_gaussian_bootstrap_filter(A, Q, H, R, N, T, y):
    '''xk|xk-1 = N(A, Q)
        yk|xk = N(H,R)
        T: time step
        y: measurement'''

    # Initialization
    n = A.shape[0]
    w_record = []
    x_record = []
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
        x_record.append(prev_x)
    return x_record, w_record, m_final, var
