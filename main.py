import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_DIR = "./dataset"

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
    
    plot_pca(test_normal_pca)
    

    print("[INFO] Datasets loaded successfully.")
    summarize("Training (normal)", train)
    summarize("Testing (normal)", test_normal)
    summarize("Testing (attack)", test_attack)