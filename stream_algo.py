import numpy as np
from sklearn.metrics import accuracy_score
from core import (linear_kernel, margin_stream, bookkeeping,
                  build_R, expand_R, compute_beta, compute_gamma, mutual_information_proxy, Q_factory)

def incremental_mi_streaming(X, y, C=1.0, target_accuracy=0.90, min_delta=1e-6):
    accepted_candidates = []
    rejected_candidates = []
    n_samples = X.shape[0]
    no_progress_counter = 0
    max_no_progress = 100 

    #### NUOVO: Q dinamico
    Q_func = Q_factory(X, y)

    a = np.zeros(n_samples)
    b = 0.0

    # Margini iniziali negativi
    g = [-1 for _ in range(n_samples)]

    S, E, R = bookkeeping(a, g, C)
    R_matrix = None

    accuracy_progress = []
    sv_progress = []
    iterazioni = []

    # Primo punto (bootstrap)
    first_candidate = 0
    gc = -1
    qcc = Q_func(first_candidate, first_candidate)
    delta_ac = min(-gc / qcc, C)
    a[first_candidate] += delta_ac
    b -= y[first_candidate] * delta_ac

    g = [margin_stream(X[i], a[a > 0], b, X[a > 0]) for i in range(n_samples)]
    S, E, R = bookkeeping(a, g, C)
    R_matrix = build_R(S, y, X, Q_func)

    # STREAMING
    for i in range(1, n_samples):

        g = [margin_stream(X[j], a[a > 0], b, X[a > 0]) for j in range(i + 1)]
        R_candidates = [j for j in range(i + 1)]

        if not R_candidates:
            continue

        candidate = max(R_candidates, key=lambda idx: mutual_information_proxy(idx, g))
        gc = g[candidate]

        if gc > -0.05:
            continue

        # Espandi R e calcola beta/gamma
        R_new = expand_R(R_matrix, S, y, X, candidate, Q_func)
        S_with_candidate = S.union({candidate})
        beta = compute_beta(R_new, S_with_candidate, y, X, candidate, Q_func)
        if beta is None:
            print(f"Iter {i}: Matr. R singolare, salto il candidato {candidate}")
            continue

        gamma = compute_gamma(S, candidate, beta, y, X, Q_func)

        if gamma[candidate] > 0:
            delta_ac = min(-gc / gamma[candidate], C - a[candidate])
        else:
            delta_ac = 0.0

        if abs(delta_ac) < min_delta:
            if i < 200 or i % 100 == 0:
                print(f"Iter {i}: delta_ac troppo piccolo ({delta_ac:.8f}), imposto delta minimo 1e-3")
            delta_ac = np.sign(delta_ac) * 1e-3
            no_progress_counter += 1
        else:
            no_progress_counter = 0 

        if no_progress_counter >= max_no_progress:
            print(f"\nEarly stopping: nessun progresso significativo per {max_no_progress} candidati consecutivi.")
            break

        if delta_ac == 0.0:
            continue

        # Calcola il delta_a proposto
        delta_a = np.zeros_like(a)
        S_list = list(S)
        for j, s in enumerate(S_list):
            delta_a[s] = beta[j + 1] * delta_ac
        delta_a[candidate] += delta_ac

        # --- SAFE UPDATE ---
        a_temp = a.copy()
        b_temp = b

        a_new = a + delta_a
        b_new = b + beta[0] * delta_ac

        g_new = [margin_stream(X[j], a_new[a_new > 0], b_new, X[a_new > 0]) for j in range(n_samples)]
        y_pred_new = np.sign([margin_stream(X[j], a_new[a_new > 0], b_new, X[a_new > 0]) for j in range(n_samples)])
        acc_new = accuracy_score(y, y_pred_new)

        y_pred_old = np.sign([margin_stream(X[j], a_temp[a_temp > 0], b_temp, X[a_temp > 0]) for j in range(n_samples)])
        acc_old = accuracy_score(y, y_pred_old)

        epsilon_acc = 0.01

        if acc_new >= acc_old - epsilon_acc:
            a = a_new
            b = b_new
            g = g_new
            S_new, E, R = bookkeeping(a, g, C)
            acc = acc_new

            if acc_new >= acc_old:
                print(f"Iter {i}: Update accettato. Accuracy migliorata o uguale {acc_old:.4f} → {acc_new:.4f}, delta_ac={delta_ac:.6f}")
            else:
                print(f"Iter {i}: Update accettato con peggioramento controllato {acc_old:.4f} → {acc_new:.4f}, delta_ac={delta_ac:.6f}")

            accepted_candidates.append(candidate)
        else:
            print(f"Iter {i}: Update SCARTATO. Accuracy peggiorava troppo ({acc_old:.4f} → {acc_new:.4f})")
            rejected_candidates.append(candidate)
            continue

        accuracy_progress.append(acc)
        sv_progress.append(len(S))
        iterazioni.append(i)

        # Aggiorna matrice R
        if S_new != S:
            S = S_new
            R_matrix = build_R(S, y, X, Q_func)
        else:
            R_matrix = R_new

        print(f"Iterazione {i}: Accuracy {acc:.4f} | SV {len(S)} | Candidato {candidate}")
        print(f"Iter {i}: gc={gc:.4f}, gamma={gamma[candidate]:.6f}, delta_ac={delta_ac:.6f}, SV={len(S)}")

    return accuracy_progress, sv_progress, iterazioni, accepted_candidates, rejected_candidates
