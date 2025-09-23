# step1_load.py
"""
Loads data into NumPy arrays and summarizes their statistics.
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_DIR = "./dataset"

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Load each dataset as float32
    train = np.load(os.path.join(DATA_DIR, "training_normal.npy")).astype(np.float64)
    test_normal = np.load(os.path.join(DATA_DIR, "testing_normal.npy")).astype(np.float64)
    test_attack = np.load(os.path.join(DATA_DIR, "testing_attack.npy")).astype(np.float64)
    return train, test_normal, test_attack

def summarize(name, arr):
    print(f"\n{name}:")
    print(f"  shape = {arr.shape}, dtype = {arr.dtype}")
    print(f"  mean = {np.mean(arr):.6g}, std = {np.std(arr):.6g}")
    print(f"  min = {np.min(arr):.6g}, max = {np.max(arr):.6g}")
    print(f"  nan_count = {np.isnan(arr).sum()}")


def plot_pca(transformed_data):
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
    train, test_normal, test_attack = load_dataset()
    scaler = StandardScaler()
    model = PCA(n_components=2)
    model.fit(train)
    test_normal_pca = model.transform(test_normal)
    test_attack_pca = model.transform(test_attack)
    
    plot_pca(test_attack_pca)
    

    print("[INFO] Datasets loaded successfully.")
    summarize("Training (normal)", train)
    summarize("Testing (normal)", test_normal)
    summarize("Testing (attack)", test_attack)