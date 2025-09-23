# step1_load.py
"""
Step 1: Dataset loading for Assignment 2 - Anomaly Detection.

We have three files in ./dataset/:
    - training_normal.npy
    - testing_normal.npy
    - testing_attack.npy

This script loads them into numpy arrays, checks their shapes,
and prints quick summaries. Arrays are cast to float32 so they
are ready for PCA and clustering in later steps.
"""

import numpy as np
import os

DATA_DIR = "./dataset"

def load_dataset():
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

if __name__ == "__main__":
    train, test_normal, test_attack = load_dataset()

    print("[INFO] Datasets loaded successfully.")
    summarize("Training (normal)", train)
    summarize("Testing (normal)", test_normal)
    summarize("Testing (attack)", test_attack)

    print("\n[Next step hint]")
    print(" - You will FIT PCA only on the Training (normal) data.")
    print(" - Then use that fitted PCA to TRANSFORM both testing sets.")
