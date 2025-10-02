# CAML Assignment 2

This repository contains the code and resources for Assignment 2

## Contents
- Source code for PCA, K-Means, and DBSCAN tasks
- Dataset files (.npy format)
- Instructions and documentation

## PCA
- Reduces the dimensionality of the KDD99 dataset from 41 features down to 2 components for visualization
- Run MainTest.py
- Displays a scatter plot of training normal, testing normal, and testing attack samples

## K-Means 
- Clusters network traffic and uses distance-to-centroid with a 95th percentile threshold to detect anomalies
- Parameters are set inside mainTest.py (k=4, maxIters=100, etc.)
- Run mainTest.py using Python3. Our experiments were done with Python 3.13.7
- Outputs confusion matrix, Accuracy, TPR, FPR, and F1-score

## DBSCAN
- Clusters network traffic using density-based clustering. Tested with different values of eps and minSamples
- Parameters (eps, minSamples) can be modified in mainTest.py
- Run MainTest.py
- Outputs confusion matrices and performance metrics for each parameter combination

## Dependencies
- NumPy
- Matplotlib
- Scikit-learn

