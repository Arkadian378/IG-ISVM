from sklearn.datasets import fetch_covtype
from sklearn.utils import shuffle
from stream_algo import incremental_mi_streaming
import matplotlib.pyplot as plt
from plot_utils import plot_candidate_decisions

#########################################
####### DATI : Covertype ################
#########################################

print("Caricamento dataset Covertype...")
X_full, y_full = fetch_covtype(return_X_y=True)
y_full = (y_full == 2).astype(int)  
y_full = 2 * y_full - 1  # -1 e +1

X_stream, y_stream = shuffle(X_full[:10000], y_full[:10000], random_state=42)

#########################################
####### STREAMING #######################
#########################################

accuracy_progress, sv_progress, iterazioni, accepted, rejected = incremental_mi_streaming(
    X_stream, y_stream, C=1.0
)

#########################################
####### RISULTATI #######################
#########################################

if len(accuracy_progress) > 0:
    print(f"\nAccuracy finale: {accuracy_progress[-1]:.4f}")
    print(f"Support Vectors finali: {sv_progress[-1]}")
else:
    print("\nNessun aggiornamento effettuato.")

#########################################
####### GRAFICI #########################
#########################################

plt.figure(figsize=(8,5))
plt.plot(iterazioni, accuracy_progress, marker='o')
plt.xlabel('Numero di punti processati (stream)')
plt.ylabel('Accuracy')
plt.title('Accuracy nel tempo (Streaming Covertype)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(iterazioni, sv_progress, marker='s')
plt.xlabel('Numero di punti processati (stream)')
plt.ylabel('Numero di Support Vectors')
plt.title('Support Vectors nel tempo (Streaming Covertype)')
plt.grid(True)
plt.show()

plot_candidate_decisions(X_stream, y_stream, accepted, rejected)
