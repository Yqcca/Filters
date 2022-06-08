import numpy as np


#First Order EKF Implementation
def predict(f, F, Q, m_t):
    predicted_m = f(m_t)
    predicted_P = F(m_t) @ predicted_P @ np.transpose(F(m_t)) + Q
    return predicted_m, predicted_P

def update(h, H, R, y, predicted_m, predicted_P):
    v = y - h(predicted_m)
    S = H(predicted_m) @ predicted_P @ np.transpose(H(predicted_m)) + R
    K = predicted_P @ np.transpose(H(predicted_m)) @ np.linalg.inv(S)
    updated_m = predicted_m + K @ v
    updated_P = predicted_P - K @ S @ np.transpose(K)
    return updated_m, updated_P
