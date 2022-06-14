import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
#Following Algorithm 5.14 Sarka

def predict(f, m, P, Q, alpha, beta, kappa, dim):
    points = MerweScaledSigmaPoints(n=dim, alpha=alpha, beta=beta, kappa=kappa)
    sigmas = points.sigma_points(m, P)
    # Pass sigmas through the dynamic model
    f_sigmas = np.empty((dim*2+1, dim))
    for i in range(dim*2+1):
        f_sigmas[i] = f(np.array([[sigmas[i, k] for k in range(dim)]]).T).T
    # check points.wm and f_sigmas correct dims
    predicted_m = np.array([np.sum(points.Wc * f_sigmas.T, axis=0)]).T
    res_1 = points.Wc * (f_sigmas - predicted_m).T
    res_2 = np.transpose(f_sigmas - predicted_m)
    res_3 = res_1.T @ res_2
    res_4 = (points.Wc * (f_sigmas - predicted_m).T).T @ np.transpose(f_sigmas - predicted_m)
    predicted_P = np.sum((points.Wc * (f_sigmas - predicted_m).T).T @ np.transpose(f_sigmas - predicted_m), axis = 1) + Q
    return predicted_m, predicted_P

def update(h, R, y, predicted_m, predicted_P, alpha, beta, kappa, dim):
    points = MerweScaledSigmaPoints(n=dim, alpha=alpha, beta=beta, kappa=kappa)
    sigmas = points.sigma_points(predicted_m, predicted_P)
    # Pass sigmas through the measurement model
    h_sigmas = np.empty(dim*2+1)
    for i in range(dim*2+1):
        h_sigmas[i] = h(np.array([[sigmas[i, k] for k in range(dim)]]).T)
    mu_k = sum(points.Wm * h_sigmas)
    S = sum(points.Wc * (h_sigmas - mu_k) @ np.transpose(h_sigmas - mu_k)) + R
    C = sum(points.Wc * (sigmas - predicted_m) @ np.transpose(h_sigmas - mu_k))
    K = C @ np.linalg.inv(S)
    update_m = predicted_m + K @ (y - mu_k)
    update_P = predicted_P - K @ S @ np.transpose(K)
    return update_m, predicted_P


    