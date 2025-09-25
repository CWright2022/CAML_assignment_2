from mimetypes import init
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score

DATA_DIR = "./dataset"

class KMeans:
    def __init__(self, k_clusters=8, threshold=0.0001):
        self.k_clusters = int(k_clusters)
        self.init = init
        self.n_init = 10
        self.max_iter = 300
        self.threshold = float(threshold)

        # Attributes set after fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def _init_centroids(self, X, rng):
        n_samples, n_features = X.shape
        k = self.k_clusters

        # choose k distinct points at random
        idx = rng.choice(n_samples, size=k, replace=False)
        centers = X[idx].astype(float).copy()
        return centers

    def _compute_distances(self, X, centers):
        """
        Return Euclidean distances (n_samples, k).
        Uses broadcasting; for large datasets you might want more memory-efficient code.
        """
        # shape: (n_samples, k)
        return np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    def fit(self, X):
        """
        Fit k-means to data X (shape: n_samples x n_features).
        Uses self.n_init restarts and keeps the solution with smallest inertia.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        rng = np.random.RandomState()

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

        for init_no in range(self.n_init):
            centers = self._init_centroids(X, rng)

            for iteration in range(self.max_iter):
                # assignment step
                dists = self._compute_distances(X, centers)
                labels = np.argmin(dists, axis=1)

                # update step
                new_centers = np.zeros_like(centers)
                for j in range(self.k_clusters):
                    members = X[labels == j]
                    if members.shape[0] == 0:
                        # empty cluster -> reinitialize centroid to random point
                        new_centers[j] = X[rng.randint(n_samples)]
                    else:
                        new_centers[j] = members.mean(axis=0)

                # check convergence (max centroid movement)
                center_shifts = np.linalg.norm(new_centers - centers, axis=1)
                centers = new_centers
                if center_shifts.max() <= self.threshold:
                    print(f"[init {init_no}] Converged at iter {iteration}")
                    break

            # compute inertia (sum of squared distances to assigned centroid)
            dists = self._compute_distances(X, centers)
            labels = np.argmin(dists, axis=1)
            closest_dists = dists[np.arange(n_samples), labels]
            inertia = np.sum(closest_dists ** 2)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

            print(f"[init {init_no}] inertia={inertia:.6g}")

        # store best result
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        """
        Assign labels to X using learned cluster centers.
        """
        X = np.asarray(X, dtype=float)
        dists = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def transform(self, X):
        """
        Return distances from each sample to each centroid.
        shape -> (n_samples, n_clusters)
        """
        X = np.asarray(X, dtype=float)
        return self._compute_distances(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Loads the dataset(s) from the filenames in DATA_DIR. Returns 3 numpy arrays:
    '''
    train = np.load(os.path.join(DATA_DIR, "training_normal.npy")).astype(np.float64)
    test_normal = np.load(os.path.join(DATA_DIR, "testing_normal.npy")).astype(np.float64)
    test_attack = np.load(os.path.join(DATA_DIR, "testing_attack.npy")).astype(np.float64)
    return train, test_normal, test_attack

def summarize(name, arr):
    '''
    summarizes a numpy array with basic statistics
    '''
    print(f"\n{name}:")
    print(f"  shape = {arr.shape}, dtype = {arr.dtype}")
    print(f"  mean = {np.mean(arr):.6g}, std = {np.std(arr):.6g}")
    print(f"  min = {np.min(arr):.6g}, max = {np.max(arr):.6g}")
    print(f"  nan_count = {np.isnan(arr).sum()}")


def plot_pca(transformed_data):
    '''
    plots PCA transformed data in 2D
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1],
                c="blue", alpha=0.5, label="Training (normal)")

    plt.title("PCA projection of network traffic data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # load datasets
    train, test_normal, test_attack = load_dataset()
    # create PCA model, fit to training data
    model = PCA(n_components=2)
    model.fit(train)
    #apply model to datasets (make them 2d)
    test_normal_pca = model.transform(test_normal)
    test_attack_pca = model.transform(test_attack)
    train_pca = model.transform(train)
    
    plot_pca(test_normal_pca)
    

    print("[INFO] Datasets loaded successfully.")
    summarize("Training (normal)", train)
    summarize("Testing (normal)", test_normal)
    summarize("Testing (attack)", test_attack)
    
    k = 4
    kmeans = KMeans(k_clusters=k, threshold=0.0001)
    kmeans.fit(train_pca)
    
    
    train_dists = kmeans.transform(train_pca).min(axis=1)
    test_normal_dists = kmeans.transform(test_normal_pca).min(axis=1)
    test_attack_dists = kmeans.transform(test_attack_pca).min(axis=1)

    # Threshold chosen from training distribution (e.g., 95th percentile)
    threshold = np.percentile(train_dists, 95)

    # Apply threshold to test sets
    y_true = np.array([0] * len(test_normal_dists) + [1] * len(test_attack_dists))  # 0=normal, 1=attack
    y_scores = np.concatenate([test_normal_dists, test_attack_dists])               # continuous anomaly scores
    y_pred = (y_scores > threshold).astype(int)                                    # binary predictions

    # --- Step 5: metrics ---
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Normal", "Attack"], digits=4)
    auc = roc_auc_score(y_true, y_scores)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall for attack class
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False positive rate
    f1 = f1_score(y_true, y_pred)

    # Print results
    print("[INFO] KMeans anomaly detection results")
    print(f"  PCA components: 2")
    print(f"  KMeans clusters: {k}")
    print(f"  Threshold (train 95th percentile): {threshold:.4f}\n")

    print("Confusion Matrix:")
    print(cm)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  TPR: {tpr:.4f}")
    print(f"  FPR: {fpr:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print("\nClassification Report (Precision, Recall, F1):")
    print(report)
    print(f"ROC AUC (using distance scores): {auc:.4f}")
    