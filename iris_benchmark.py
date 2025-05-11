from sklearn.datasets import load_iris
import numpy as np
from core import Q_factory, margin_batch
from algos import batch_svm, incremental_random, incremental_mi
from sklearn.metrics import accuracy_score
from plot_utils import (plot_accuracy_mi, plot_sv_count, plot_final_accuracy,
                        plot_pca_predictions, plot_learning_curves, plot_accuracy_vs_sv)

##############################
##### DATA - IRIS ###########
##############################

iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]
y = 2 * y - 1 

C = 1.0

##############################
##### SVM BATCH BASELINE #####
##############################

batch_acc, batch_time = batch_svm(X, y, C)
print("\nSVM batch:")
print(f"Accuracy: {batch_acc:.4f} | Tempo: {batch_time:.4f}s")

##############################
##### INCREMENTAL RANDOM #####
##############################

import time

start = time.time()
a_random, b_random, acc_random, it_random = incremental_random(X, y, C)
random_time = time.time() - start

Q_func = Q_factory(X, y)

y_pred_random = np.sign([margin_batch(i, a_random, b_random, Q_func, y) for i in range(len(y))])
random_acc = accuracy_score(y, y_pred_random)

print("\nIncremental SVM Random:")
print(f"Accuracy: {random_acc:.4f} | Iterazioni: {it_random} | Tempo: {random_time:.4f}s")

##############################
##### INCREMENTAL MI #########
##############################

start = time.time()
a_mi, b_mi, acc_mi, sv_count, it_mi = incremental_mi(X, y, C)
mi_time = time.time() - start

y_pred_mi = np.sign([margin_batch(i, a_mi, b_mi, Q_func, y) for i in range(len(y))])
mi_acc = accuracy_score(y, y_pred_mi)

print("\nIncremental SVM con Mutual Information:")
print(f"Accuracy: {mi_acc:.4f} | Iterazioni: {it_mi} | Tempo: {mi_time:.4f}s")

##############################
########## PLOTTING ##########
##############################

plot_accuracy_mi(acc_mi)
plot_sv_count(sv_count)
plot_final_accuracy(batch_acc, random_acc, mi_acc)
plot_pca_predictions(X, y_pred_mi, y)
plot_learning_curves(acc_random, acc_mi)
plot_accuracy_vs_sv(acc_mi, sv_count, method_label="Incremental MI (Iris)")
