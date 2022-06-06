import numpy as np
from scipy import stats

# Linear Gaussian case
def prior_sample(U, Q, N):
    '''
    Initial sample from the prior distribution.

    Parameters
    ----------
    U : arr
        Mean of prior distribution.
    
    Q : arr
        Covariance of prior distribution.
    
    N : int
        Number of samples.

    Returns
    ----------
    x_0 : arr
        An initialized numpy array samples.
    w_0 : arr
        An initialized numpy array weights.
    '''

    x_0 = np.random.multivariate_normal(mean=U, cov=Q, size = N)
    w_0 = np.array([1/N for i in range(N)])
    return x_0, w_0

def normalized_weight(w):

    '''
    Compute the normalized weights given a measurement model and a prior.

    Parameters
    ----------
    w : arr
        An N-length numpy array unnormalized weights.

    Returns
    ----------
    v : arr
        An N-length numpy array normalized weights.
    '''
    v = []
    s = np.sum(w)
    for i in range(len(w)):
        v.append(w[i]/s)
    return np.array(v)

def resampling(w, x):
    '''
    Derive the resampling of samples based on their weights.

    Parameters
    ----------
    w : arr
        An numpy array normalized weights.
    
    x : arr
        An numpy array samples.

    Returns
    ----------
    m : arr
        An numpy array constant weights.
    n : arr
        An numpy array resampled samples.
    '''
    ind = [i for i in range(len(w))]
    n = np.zeros(x.shape)
    for i in range(len(w)):
        n[i] = x[np.random.choice(ind, size = 1, p = w)]
    m = np.zeros(len(w))
    for i in range(len(w)):
        m[i] = 1/len(w)
    return m, n

def check_resampling(w, x):
    '''
    Check whether need to do resampling and return the adjusted weights and samples.

    Parameters
    ----------
    w : arr
        An numpy array normalized weights.
    
    x : arr
        An numpy array samples.

    Returns
    ----------
    m : arr
        An numpy array adjusted weights.
    n : arr
        An numpy array adjusted samples.
    '''

    rneff = 0
    for i in range(len(w)):
        rneff += w[i]**2
    neff = 1/rneff
    if neff < len(w)/10:
        return resampling(w, x)
    else:
        return w, x

def linear_gaussian_importance_distribution(prev_x, y, A, Q):
    '''
    Derive the linear gaussian importance distribution.

    Parameters
    ----------
    prev_x : float
        A sample of x at the previous time-step.
    
    y : float
        The measurement of x.

    A: arr
        The transition matrix of the dynamic model.

    Q : arr
        The process noise.

    Returns
    ----------
    f : scipy.stats._multivariate.multivariate_normal_gen
        The estimated linear gaussian importance distribution.
    '''

    while prev_x.shape != y.shape:
        y = np.append(y,0)
    m = 0.5*A@prev_x + 0.5*y
    f = stats.multivariate_normal(mean=m, cov=Q)
    return f

def linear_gaussian_particle_filter(A, Q, H, R, N, T, y):
    '''
    Derive the linear gaussian importance distribution.

    Parameters
    ----------
    A : arr
        The transition matrix of the dynamic model.
    
    Q : arr
        The process noise.

    H : arr
        The measurement model matrix.
    
    R : arr
        The measurement noise.

    N : int
        The number of samples.

    T : int
        The number of time steps.

    y : arr
        The T-length numpy array of measurement.

    Returns
    ----------
    m_final : arr
        An numpy array of filtered dynamic states.
    '''

    # Initialization
    n = A.shape[0]
    prev_x, prev_w = prior_sample(np.zeros(n), Q, N)
    m_final = np.zeros((T, n))
    for i in range(N):
        m_final[0] += prev_w[i] * prev_x[i]

    # Calculate weight for each time step
    for k in range(1, T):
        x = np.zeros((N, n))
        w = np.zeros(N)
        for i in range(N):
            f = linear_gaussian_importance_distribution(prev_x[i], y[k], A, Q)
            x[i] = np.array(f.rvs(size=1))
            g1 = stats.multivariate_normal(mean=H@x[i], cov=R)
            g2 = stats.multivariate_normal(mean=A@prev_x[i], cov=Q)
            w[i] = (prev_w[i]*g1.pdf(y[k])*g2.pdf(x[i])/f.pdf(x[i])) + np.finfo(float).eps
        w = normalized_weight(w)
        w, x = check_resampling(w, x)
        for i in range(N):
            m_final[k] += w[i] * x[i]
        prev_x = x
        prev_w = w
    return m_final

def linear_gaussian_bootstrap_filter(A, Q, H, R, N, T, y):
    '''
    Derive the linear gaussian importance distribution.

    Parameters
    ----------
    A : arr
        The transition matrix of the dynamic model.
    
    Q : arr
        The process noise.

    H : arr
        The measurement model matrix.
    
    R : arr
        The measurement noise.

    N : int
        The number of samples.

    T : int
        The number of time steps.

    y : arr
        The T-length numpy array of measurement.

    Returns
    ----------
    m_final : arr
        An numpy array of filtered dynamic states.
    '''

    # Initialization
    n = A.shape[0]
    prev_x = np.random.multivariate_normal(mean=np.zeros(n), cov=Q, size = N)
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
        for i in range(N):
            m_final[k] += w[i] * x[i]
        prev_x = x
    return m_final

def linear_gaussian_rb_particle_filter(A, Q, H, R, N, T, y):
    '''
    Derive the linear gaussian importance distribution.

    Parameters
    ----------
    A : arr
        The transition matrix of the dynamic model.
    
    Q : arr
        The process noise.

    H : arr
        The measurement model matrix.
    
    R : arr
        The measurement noise.

    N : int
        The number of samples.

    T : int
        The number of time steps.

    y : arr
        The T-length numpy array of measurement.

    Returns
    ----------
    m_final : arr
        An numpy array of filtered dynamic states.
    '''
    pass