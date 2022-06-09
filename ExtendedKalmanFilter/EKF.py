import numpy as np


#First Order EKF Implementation
def predict(f, F, Q, m_t, P_t):
    predicted_m = f(m_t)
    predicted_P = F(m_t) @ P_t @ np.transpose(F(m_t)) + Q
    return predicted_m, predicted_P

def update(h, H, R, y, predicted_m, predicted_P):
    v = y - h(predicted_m)
    S = H(predicted_m) @ predicted_P @ np.transpose(H(predicted_m)) + R
    if S.ndim == 1:
        S_inv = np.array([1/S[0]])
    else:
        S_inv = np.linalg.inv(S)
    
    #Below since tranpose of 1D array 
    if (predicted_P @ np.transpose(H(predicted_m))).ndim == 1:
        K_res_1 = (predicted_P @ np.transpose(H(predicted_m)))[np.newaxis]
        K_res_1 = K_res_1.T
    else:
        K_res_1 = predicted_P @ np.transpose(H(predicted_m))

    K = K_res_1 @ S_inv

    if (predicted_m + K).ndim == 1:
        updated_m_res_1 = (predicted_m + K)[np.newaxis]
        updated_m_res_1 = updated_m_res_1.T
    else:
        updated_m_res_1 = predicted_m + K
    updated_m = updated_m_res_1 @ v

    if K.ndim == 1:
        updated_res_1 = K[np.newaxis].T
        t = K[np.newaxis].T @ S
        res = t[np.newaxis].T
        updated_res_2 = res @ K[np.newaxis]
    else:
        updated_res_2 = K @ S @ np.transpose(K)

    updated_P = predicted_P - updated_res_2
    return updated_m, updated_P
