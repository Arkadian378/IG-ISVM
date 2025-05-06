from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from stream_algo import incremental_mi_streaming  # Usa la funzione aggiornata!
import matplotlib.pyplot as plt
from plot_utils import plot_candidate_decisions
from core import Q_factory, margin_stream

#########################################
####### DATI : IRIS #####################
#########################################

iris = load_iris()
X_full = iris.data[iris.target != 2]
y_full = iris.target[iris.target != 2]
y_full = 2 * y_full - 1  # 0 -> -1, 1 -> +1

# Shuffle per simulare streaming
X_stream, y_stream = shuffle(X_full, y_full, random_state=42)

#########################################
####### APPRENDIMENTO IN STREAMING ######
#########################################

accuracy_progress, sv_progress, iterazioni, accepted, rejected = incremental_mi_streaming(
    X_stream, y_stream, C=1.0, target_accuracy=0.90, min_delta=1e-6
)

#########################################
####### RISULTATI #######################
#########################################

if len(accuracy_progress) > 0:
    print(f"\nAccuracy finale: {accuracy_progress[-1]:.4f}")
    print(f"Support Vectors finali: {sv_progress[-1]}")
    print(f"Totale candidati accettati: {len(accepted)}")
    print(f"Totale candidati scartati: {len(rejected)}")
else:
    print("\nNessun aggiornamento effettuato.")

#########################################
####### GRAFICI #########################
#########################################

plt.figure(figsize=(8,5))
plt.plot(iterazioni, accuracy_progress, marker='o', label='Accuracy')
plt.axhline(y=0.90, color='r', linestyle='--', label='Target accuracy')
plt.xlabel('Numero di punti processati (stream)')
plt.ylabel('Accuracy')
plt.title('Accuracy nel tempo (Streaming Iris)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(iterazioni, sv_progress, marker='s', color='purple')
plt.xlabel('Numero di punti processati (stream)')
plt.ylabel('Numero di Support Vectors')
plt.title('Support Vectors nel tempo (Streaming Iris)')
plt.grid(True)
plt.show()

plot_candidate_decisions(X_stream, y_stream, accepted, rejected)
