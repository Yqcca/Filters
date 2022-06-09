import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
#Following Algorithm 5.14 Sarka

def predict(f, m, P, Q, alpha, beta, kappa, dim):
    points = MerweScaledSigmaPoints(n=dim, alpha=alpha, beta=beta, kappa=kappa)
    sigmas = points.sigma_points(m, P)
    # Pass sigmas through the dynamic model
    f_sigmas = np.empty(dim*2+1)
    for i in range(dim*2+1):
        f_sigmas[i] = f(np.array([[sigmas[i, k] for k in range(dim)]]).T)
    # check points.wm and f_sigmas correct dims
    predicted_m = sum(points.Wm * f_sigmas)
    predicted_P = sum(points.Wc * (f_sigmas - predicted_m) @ np.transpose(f_sigmas - predicted_m)) + Q
    return predicted_m, predicted_P

def update(h, R, predicted_m, predicted_P, alpha, beta, kappa, dim):
    points = MerweScaledSigmaPoints(n=dim, alpha=alpha, beta=beta, kappa=kappa)
    sigmas = points.sigma_points(predicted_m, predicted_P)
    # Pass sigmas through the measurement model
    h_sigmas = np.empty(dim*2+1)
    for i in range(dim*2+1):
        h_sigmas[i] = h(np.array([[sigmas[i, k] for k in range(dim)]]).T)
    mu_k = sum(points.Wm * h_sigmas)
    S_k = sum(points.Wc * (h_sigmas - mu_k) @ np.transpose(h_sigmas - mu_k)) + R
    C_k = sum(points.Wc * (sigmas - predicted_m) @ np.transpose(h_sigmas - mu_k)) 
    