import numpy as np
import numpy.linalg as la

#######################################
####### Q-on-demand e Q factory ########
#######################################

def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

def Q_ondemand(i, j, X, y):
    return y[i] * y[j] * np.dot(X[i], X[j])

def Q_factory(X, y):
    n_samples = X.shape[0]
    if n_samples <= 1000:
        # Dataset piccolo → usa Q completo
        K = linear_kernel(X, X)
        Q_matrix = (y[:, None] * y[None, :]) * K
        return lambda i, j: Q_matrix[i, j]
    else:
        # Dataset grande → usa Q-on-demand
        return lambda i, j: Q_ondemand(i, j, X, y)

#######################################
####### MARGINE BATCH #################

def margin_batch(i, a, b, Q_func, y):
    return sum(
        a[j] * Q_func(i, j)
        for j in range(len(a))
    ) + y[i] * b - 1

#######################################
####### MARGINE STREAMING #############

def margin_stream(x_new, a, b, X_sv):
    if len(a) == 0:
        return -1
    k = np.dot(X_sv, x_new)
    return np.dot(a, k) + b - 1

#######################################
####### BOOKKEEPING ###################

def bookkeeping(a, g, C):
    S, E, R = set(), set(), set()
    for i in range(len(a)):
        if (0 < a[i] < C) or np.isclose(g[i], 0.0, atol=1e-6):
            S.add(i)
        elif a[i] >= C - 1e-8:
            E.add(i)
        else:
            R.add(i)
    return S, E, R

#######################################
####### BUILD R #######################

def build_R(S, y, X, Q_func):
    k = len(S)
    R = np.zeros((k + 1, k + 1))
    S_list = list(S)
    for i, s_i in enumerate(S_list):
        R[0, i + 1] = y[s_i]
        R[i + 1, 0] = y[s_i]
    for i, s_i in enumerate(S_list):
        for j, s_j in enumerate(S_list):
            R[i + 1, j + 1] = Q_func(s_i, s_j)
    return R

#######################################
####### EXPAND R ######################

def expand_R(R_old, S, y, X, candidate, Q_func):
    S_list = list(S)
    k = len(S_list)

    R_new = np.zeros((k + 2, k + 2))

    if R_old is not None and R_old.shape[0] == (k + 1):
        R_new[:k + 1, :k + 1] = R_old
    else:
        R_new[:k + 1, :k + 1] = build_R(S, y, X, Q_func)

    y_new = y[candidate]
    R_new[0, k + 1] = y_new
    R_new[k + 1, 0] = y_new

    for i, s in enumerate(S_list):
        R_new[i + 1, k + 1] = Q_func(s, candidate)
        R_new[k + 1, i + 1] = Q_func(s, candidate)

    R_new[k + 1, k + 1] = Q_func(candidate, candidate)

    return R_new

#######################################
####### COMPUTE BETA ##################

def compute_beta(R_matrix, S, y, X, candidate, Q_func):
    S_list = list(S)
    k = len(S_list)

    v = np.zeros(R_matrix.shape[0])
    v[0] = 1

    if len(v) == k + 2:
        for i, s in enumerate(S_list):
            v[i + 1] = Q_func(s, candidate)
        v[-1] = Q_func(candidate, candidate)
    elif len(v) == k + 1:
        for i, s in enumerate(S_list):
            v[i + 1] = Q_func(s, candidate)
    else:
        return None

    try:
        epsilon = 1e-8
        R_matrix_reg = R_matrix + epsilon * np.eye(R_matrix.shape[0])
        beta = la.solve(R_matrix_reg, v)
    except la.LinAlgError:
        return None

    return beta

#######################################
####### COMPUTE GAMMA #################

def compute_gamma(S, candidate, beta, y, X, Q_func):
    if beta is None:
        return None

    n_samples = X.shape[0]
    gamma = np.zeros(n_samples)
    S_list = list(S)

    for i in range(n_samples):
        gamma[i] = Q_func(i, candidate)
        for j, s in enumerate(S_list):
            gamma[i] += Q_func(i, s) * beta[j + 1]
        gamma[i] += y[i] * beta[0]

    return gamma

#######################################
####### MUTUAL INFORMATION PROXY ######

def mutual_information_proxy(i, g):
    g_clip = np.clip(g[i], -100, 100)
    p = 1 / (1 + np.exp(-g_clip))
    return -np.log(max(min(p, 1 - 1e-8), 1e-8))
