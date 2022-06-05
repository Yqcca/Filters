import numpy as np


#See p77 of Sarka book
#Assumes all inputs are numpy arrays
def predict(A, Q, m_t, P_t):
    predicted_m = A @ m_t
    predicted_P = A @ P_t @ np.transpose(A) + Q
    return predicted_m, predicted_P


def update(H, R, y, predicted_m, predicted_P):
    v = y - H @ predicted_m  #residual mean
    S = H @ predicted_P @ np.transpose(H) + R  #residual covariance
    K = predicted_P @ np.transpose(H) @ np.linalg.inv(S)
    updated_m = predicted_m + K @ v
    updated_P = predicted_P - K @ S @ np.transpose(K)
    return updated_m, updated_P
    