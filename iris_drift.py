from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from stream_algo import incremental_mi_streaming
import matplotlib.pyplot as plt
import numpy as np

#########################################
####### DATI : IRIS + DRIFT #############
#########################################

iris = load_iris()
X_full = iris.data[iris.target != 2]
y_full = iris.target[iris.target != 2]
y_full = 2 * y_full - 1

X_stream, y_stream = shuffle(X_full, y_full, random_state=42)

# --- Introduci DRIFT ---
drift_point = len(y_stream) // 2
flip_fraction = 0.3
n_flip = int(flip_fraction * (len(y_stream) - drift_point))
flip_indices = np.random.choice(range(drift_point, len(y_stream)), size=n_flip, replace=False)
y_stream[flip_indices] *= -1  # Cambia le etichette

#########################################
####### APPRENDIMENTO IN STREAMING ######
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
plt.plot(iterazioni, accuracy_progress, marker='o', label='Accuracy')
plt.xlabel('Numero di punti processati (stream)')
plt.ylabel('Metriche')
plt.title('Metriche nel tempo (Streaming Iris con Drift)')
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
