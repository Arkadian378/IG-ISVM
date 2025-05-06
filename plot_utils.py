import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

##############################################
####### STILE ACCADEMICO SEABORN #############
##############################################

sns.set_theme(style="whitegrid")
sns.set_context("talk")  # Buono per paper e presentazioni
custom_palette = sns.color_palette("deep")

##############################################
####### PLOT ACCURACY MI #####################
##############################################

def plot_accuracy_mi(accuracy_per_iteration_mi):
    plt.figure(figsize=(8,5))
    sns.lineplot(x=range(1, len(accuracy_per_iteration_mi)+1),
                 y=accuracy_per_iteration_mi,
                 marker='o', color='green')
    plt.xlabel('Iterazioni')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterazioni (Incremental MI)')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

##############################################
####### PLOT SUPPORT VECTORS COUNT ###########
##############################################

def plot_sv_count(sv_count_per_iteration):
    plt.figure(figsize=(8,5))
    sns.lineplot(x=range(1, len(sv_count_per_iteration)+1),
                 y=sv_count_per_iteration,
                 marker='s', color='purple')
    plt.xlabel('Iterazioni')
    plt.ylabel('Support Vectors')
    plt.title('Support Vectors vs Iterazioni')
    plt.tight_layout()
    plt.show()

##############################################
####### PLOT ACCURACY FINALE #################
##############################################

def plot_final_accuracy(batch_acc, random_acc, mi_acc):
    labels = ['SVM Batch', 'Incremental Random', 'Incremental MI']
    accuracies = [batch_acc, random_acc, mi_acc]

    plt.figure(figsize=(7,5))
    sns.barplot(x=labels, y=accuracies, palette=custom_palette)
    plt.ylim(0, 1.05)
    plt.ylabel('Accuracy')
    plt.title('Accuracy finale dei metodi')
    plt.tight_layout()
    plt.show()

##############################################
####### PLOT PCA PREDICTIONS #################
##############################################

def plot_pca_predictions(X, y_pred_mi, y_true):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=X_reduced[:, 0], y=X_reduced[:, 1],
        hue=y_pred_mi, palette='coolwarm', s=80, edgecolor='k'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Predizioni Incremental SVM MI (PCA 2D)')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    sns.scatterplot(
        x=X_reduced[:, 0], y=X_reduced[:, 1],
        hue=y_true, palette='coolwarm', s=80, edgecolor='k', ax=axs[0]
    )
    axs[0].set_title('Classi Vere (PCA)')

    sns.scatterplot(
        x=X_reduced[:, 0], y=X_reduced[:, 1],
        hue=y_pred_mi, palette='coolwarm', s=80, edgecolor='k', ax=axs[1]
    )
    axs[1].set_title('Predizioni Incremental MI (PCA)')

    plt.tight_layout()
    plt.show()

##############################################
####### PLOT LEARNING CURVES #################
##############################################

def plot_learning_curves(accuracy_random, accuracy_mi):
    plt.figure(figsize=(8,5))

    sns.lineplot(x=range(1, len(accuracy_random)+1),
                 y=accuracy_random, marker='o', linestyle='--',
                 color='orange', label='Incremental Random')

    sns.lineplot(x=range(1, len(accuracy_mi)+1),
                 y=accuracy_mi, marker='s', linestyle='-',
                 color='green', label='Incremental MI')

    plt.xlabel('Iterazioni')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve: Random vs MI')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

##############################################
####### PLOT DECISIONI CANDIDATI #############
##############################################

def plot_candidate_decisions(X, y, accepted_idx, rejected_idx):
    X_reduced = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1],
                    hue=y, palette='coolwarm', s=80, alpha=0.3, edgecolor='k', legend=False)

    if accepted_idx:
        sns.scatterplot(x=X_reduced[accepted_idx, 0],
                        y=X_reduced[accepted_idx, 1],
                        color='green', s=150, marker='o', label='Accettati')

    if rejected_idx:
        sns.scatterplot(x=X_reduced[rejected_idx, 0],
                        y=X_reduced[rejected_idx, 1],
                        color='red', s=150, marker='X', label='Scartati')

    plt.title('Decisioni sui candidati (PCA 2D)')
    plt.legend()
    plt.tight_layout()
    plt.show()

##############################################
####### PLOT ACCURACY VS SUPPORT VECTORS #####
##############################################

def plot_accuracy_vs_sv(accuracy_list, sv_count_list, method_label="Incremental MI"):
    plt.figure(figsize=(8,5))
    sns.lineplot(x=sv_count_list, y=accuracy_list, marker='o', label=method_label)
    plt.xlabel('Numero di Support Vectors')
    plt.ylabel('Accuracy')
    plt.title('Trade-off Accuracy vs Numero di Support Vectors')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
