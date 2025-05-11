import time
import numpy as np
from sklearn.svm import SVC
import numpy.linalg as la
from sklearn.metrics import accuracy_score
from core import (linear_kernel, margin_batch, bookkeeping, build_R, expand_R,
                  compute_beta, compute_gamma, mutual_information_proxy, Q_factory)
from tqdm import tqdm

##########################
#### SVM batch ###########
##########################

def batch_svm(X, y, C):
    svc = SVC(kernel='linear', C=C)
    start = time.time()
    svc.fit(X, y)
    elapsed = time.time() - start
    acc = svc.score(X, y)
    return acc, elapsed

##########################
#### Incremental Random ##
##########################

def incremental_random(X, y, C, max_iterations=100):
    n_samples = len(y)
    a = np.zeros(n_samples)
    b = 0.0

    Q_func = Q_factory(X, y)

    g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
    S, E, R = bookkeeping(a, g, C)

    iterations = 0
    accuracy_per_iteration = []

    while len(R) > 0 and iterations < max_iterations:
        candidate = np.random.choice(list(R))

        gc = margin_batch(candidate, a, b, Q_func, y)
        qcc = Q_func(candidate, candidate)

        delta_ac = min(-gc / qcc, C - a[candidate]) if qcc > 0 else 0.0
        if delta_ac < 1e-8:
            R.remove(candidate)
            continue

        a[candidate] += delta_ac
        b -= y[candidate] * delta_ac

        g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
        S, E, R = bookkeeping(a, g, C)

        acc = accuracy_score(y, np.sign([margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]))
        accuracy_per_iteration.append(acc)
        iterations += 1

    return a, b, accuracy_per_iteration, iterations

##########################
#### Incremental MI ######
##########################

def incremental_mi(X, y, C, max_iterations=100, min_delta=1e-6):
    n_samples = len(y)
    a = np.zeros(n_samples)
    b = 0.0

    Q_func = Q_factory(X, y)

    g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
    S, E, R = bookkeeping(a, g, C)

    first_candidate = 0
    gc = margin_batch(first_candidate, a, b, Q_func, y)
    qcc = Q_func(first_candidate, first_candidate)

    delta_ac = min(-gc / qcc, C)
    a[first_candidate] += delta_ac
    b -= y[first_candidate] * delta_ac

    g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
    S, E, R = bookkeeping(a, g, C)
    R_matrix = build_R(S, y, X, Q_func)

    iterations = 0
    accuracy_per_iteration = []
    sv_count_per_iteration = []

    while len(R) > 0 and iterations < max_iterations:
        candidate = max(R, key=lambda i: mutual_information_proxy(i, g))

        R_new = expand_R(R_matrix, S, y, X, candidate, Q_func)
        S_with_candidate = S.union({candidate})
        beta = compute_beta(R_new, S_with_candidate, y, X, candidate, Q_func)
        if beta is None:
            R.remove(candidate)
            continue

        gamma = compute_gamma(S, candidate, beta, y, X, Q_func)
        if gamma is None:
            R.remove(candidate)
            continue

        gc = margin_batch(candidate, a, b, Q_func, y)

        if gamma[candidate] > 0:
            delta_ac = min(-gc / gamma[candidate], C - a[candidate])
        else:
            delta_ac = 0.0

        if delta_ac < min_delta:
            R.remove(candidate)
            continue

        delta_a = np.zeros_like(a)
        S_list = list(S)
        for idx, s in enumerate(S_list):
            delta_a[s] = beta[idx + 1] * delta_ac
        delta_a[candidate] += delta_ac

        a += delta_a
        b += beta[0] * delta_ac

        g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
        S_new, E, R = bookkeeping(a, g, C)

        acc = accuracy_score(y, np.sign([margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]))
        accuracy_per_iteration.append(acc)
        sv_count_per_iteration.append(len(S))

        if S_new != S:
            S = S_new
            R_matrix = build_R(S, y, X, Q_func)
        else:
            R_matrix = R_new

        iterations += 1

    return a, b, accuracy_per_iteration, sv_count_per_iteration, iterations


def incremental_mi_batch(
    X, y, C=1.0, max_iterations=100,
    min_delta=1e-6, max_sv=300, max_candidates=1000,
    epsilon_acc=0.005,  
    apply_pca=False
):
    n_samples = X.shape[0]
    a = np.zeros(n_samples)
    b = 0.0

    Q_func = Q_factory(X, y)

    g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
    S, E, R = bookkeeping(a, g, C)

    first_candidate = 0
    gc = g[first_candidate]
    delta_ac = min(-gc / Q_func(first_candidate, first_candidate), C)
    a[first_candidate] += delta_ac
    b -= y[first_candidate] * delta_ac

    g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
    S, E, R = bookkeeping(a, g, C)
    R_matrix = build_R(S, y, X, Q_func)

    accuracy_per_iteration = []
    sv_count_per_iteration = []

    acc_old = accuracy_score(y, np.sign([margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]))

    for iteration in tqdm(range(1, max_iterations + 1), desc="Incremental MI Batch"):

        if len(S) >= max_sv:
            print(f"Numero massimo di SV ({max_sv}) raggiunto.")
            break

        candidates = list(R)[:max_candidates]
        if not candidates:
            print("Nessun candidato disponibile.")
            break

        candidate = max(candidates, key=lambda i: mutual_information_proxy(i, g))

        R_new = expand_R(R_matrix, S, y, X, candidate, Q_func)
        S_with_candidate = S.union({candidate})
        beta = compute_beta(R_new, S_with_candidate, y, X, candidate, Q_func)
        if beta is None:
            print(f"Iter {iteration}: Matrice R singolare. Salto candidato {candidate}.")
            continue

        gamma = compute_gamma(S, candidate, beta, y, X, Q_func)
        if gamma is None:
            continue

        gc = margin_batch(candidate, a, b, Q_func, y)

        if gamma[candidate] > 0:
            delta_ac = min(-gc / gamma[candidate], C - a[candidate])
        else:
            delta_ac = 0.0

        if abs(delta_ac) < min_delta:
            delta_ac = np.sign(delta_ac) * 1e-4

        delta_a = np.zeros_like(a)
        S_list = list(S)
        for i, s in enumerate(S_list):
            delta_a[s] = beta[i + 1] * delta_ac
        delta_a[candidate] += delta_ac

        a_new = a + delta_a
        b_new = b + beta[0] * delta_ac

        y_pred_new = np.sign([margin_batch(i, a_new, b_new, Q_func, y) for i in range(n_samples)])
        acc_new = accuracy_score(y, y_pred_new)

        if acc_new >= acc_old - epsilon_acc:
            a = a_new
            b = b_new
            acc_old = acc_new
            g = [margin_batch(i, a, b, Q_func, y) for i in range(n_samples)]
            S_new, E, R = bookkeeping(a, g, C)

            accuracy_per_iteration.append(acc_new)
            sv_count_per_iteration.append(len(S))

            if S_new != S:
                S = S_new
                R_matrix = build_R(S, y, X, Q_func)
            else:
                R_matrix = R_new

        else:
            print(f"Iter {iteration}: Update scartato. Accuracy peggiorava ({acc_old:.4f} → {acc_new:.4f})")

    return a, b, accuracy_per_iteration, sv_count_per_iteration, iteration
