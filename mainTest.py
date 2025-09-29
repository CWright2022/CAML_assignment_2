# python -m pip install --upgrade pip
# python -m pip install numpy matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import os

# Helper Functions

def euclideanDistance(pointOne, pointTwo):
    """
    Derives the Euclidean distance between two points.
    """
    eDistance = float(np.linalg.norm(pointOne - pointTwo))
    return eDistance

def loadDataset():
    """
    Load training, testing normal, and testing attack sets.
    """
    dataDirectory = "./dataset"
    trainNormal = np.load(os.path.join(dataDirectory, "training_normal.npy")).astype(np.float64)
    testNormal = np.load(os.path.join(dataDirectory, "testing_normal.npy")).astype(np.float64)
    testAttack = np.load(os.path.join(dataDirectory, "testing_attack.npy")).astype(np.float64)
    return trainNormal, testNormal, testAttack

def summarizeData(dataName, array):
    """
    Summarizes the dataset with basic statistics.
    """
    print(f"Summary of {dataName}:")
    print(f"Shape: {array.shape}")
    print(f"Mean: {np.mean(array):.4f}")
    print(f"Standard Deviation: {np.std(array):.4f}")
    #print(f"Minimum: {np.min(array):.4f}")
    #print(f"Maximum: {np.max(array):.4f}\n")

def evaluateResults(yTrue, yPred):
    """
    Build confusion matrix and calculate metrics.
    """
    TN, FP, FN, TP = np.bincount(2 * yTrue + yPred, minlength = 4)
    confusionMatrix = np.array([[TN, FP], [FN, TP]])

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return confusionMatrix, Accuracy, TPR, FPR, F1

# Class PCA Implementation
class PCA:
    """
    PCA using numpy eigen decomposition.
    """
    def __init__(self, nComponents):
        self.nComponents = nComponents
        self.mean = None
        self.components = None

    def fit(self, X):
        # Centers the data
        self.mean = np.mean(X, axis = 0)
        xCentered = X - self.mean

        # Computes the covariance matrix
        covarianceMatrix = np.cov(xCentered, rowvar = False)

        # Conducts Eigen decomposition
        eigenValues, eigenVectors = np.linalg.eigh(covarianceMatrix)

        # Sorts the eigenvalues and eigenvectors
        sortedIndices = np.argsort(eigenValues)[::-1]
        self.components = eigenVectors[:, sortedIndices[:self.nComponents]]

    def transform(self, X):
        xCentered = X - self.mean
        return np.dot(xCentered, self.components)
    
def plotPCA(trainNormalPCA, testNormalPCA, testAttackPCA):
    """
    Plot the PCA results.
    """
    plt.figure(figsize = (10, 7))
    plt.scatter(trainNormalPCA[:, 0], trainNormalPCA[:, 1], label = 'Training Normal', alpha = 0.5)
    plt.scatter(testNormalPCA[:, 0], testNormalPCA[:, 1], label = 'Testing Normal', alpha = 0.5)
    plt.scatter(testAttackPCA[:, 0], testAttackPCA[:, 1], label = 'Testing Attack', alpha = 0.5)
    plt.title('PCA Projection of Network Traffic Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

class KMeans:
    """
    K-Means clustering algorithm from scratch.
    Runs multiple initializations and keeps the best result.
    """

    def __init__(self, kClusters, maxIters = 100, threshold = 1e-4, initializationAttempts=10):
        self.kClusters = kClusters
        self.maxIters = maxIters
        self.threshold = threshold
        self.initAttempts = initializationAttempts
        self.centroids = None

    def fit(self, X):
        bestInertia = float("inf")
        bestCentroids = None

        for attempt in range(self.initAttempts):
            # Randomly initialize centroids
            randomIndices = np.random.choice(X.shape[0], self.kClusters, replace = False)
            centroids = X[randomIndices]

            for _ in range(self.maxIters):
                # Assigns the clusters
                distances = np.array([[euclideanDistance(x, centroid) for centroid in centroids] for x in X])
                labels = np.argmin(distances, axis=1)

                # Updates the centroids (i.e. handle empty clusters)
                newCentroids = []
                for k in range(self.kClusters):
                    if np.any(labels == k):
                        newCentroids.append(X[labels == k].mean(axis=0))
                    else:
                        # Reinitializes an empty cluster randomly
                        newCentroids.append(X[np.random.randint(0, X.shape[0])])
                newCentroids = np.array(newCentroids)

                # Checks for convergence
                if np.linalg.norm(newCentroids - centroids) < self.threshold:
                    break
                centroids = newCentroids

            # Computes the inertia (sum of squared distances to nearest centroid)
            inertia = np.sum(np.min(distances, axis=1) ** 2)

            if inertia < bestInertia:
                bestInertia = inertia
                bestCentroids = centroids

        self.centroids = bestCentroids

    def predict(self, X):
        distances = np.array([[euclideanDistance(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis = 1)

class DBscan:
    """
    Simplified DBSCAN for anomaly detection.
    Uses a grid-based approach to avoid O(n^2) neighbor search.
    Only finds core points.
    """
    def __init__(self, eps = 0.5, minSamples = 5):
        self.eps = eps
        self.minSamples = minSamples
        self.labels = None

    def fit(self, X):
        n = X.shape[0]
        self.labels = np.full(n, -1)  # -1 means noise
        clusterID = 0

        for i in range(n):
            if self.labels[i] != -1:
                continue  # Already processed and labeled
        
            distances = np.linalg.norm(X - X[i], axis = 1)
            neighbors = np.where(distances <= self.eps)[0]

            if len(neighbors) < self.minSamples:
                self.labels[i] = -1  # Mark as noise
            else:
                # Found a core point, start a new cluster
                self.labels[neighbors] = clusterID
                clusterID += 1

            # Find neighbors
            distances = np.linalg.norm(X - X[i], axis=1)
            neighbors = np.where(distances <= self.eps)[0]

            if len(neighbors) < self.minSamples:
                self.labels[i] = -1  # Mark as noise
            else:
                # Found a core point, start a new cluster
                self.labels[neighbors] = clusterID
                clusterID += 1

    def predict(self, X):
        return self.labels


if __name__ == "__main__":
    # Load datasets
    trainNormal, testNormal, testAttack = loadDataset()

    # Summarizes datasets
    summarizeData("Training Normal", trainNormal)
    summarizeData("Testing Normal", testNormal)
    summarizeData("Testing Attack", testAttack)

    # Applies the PCA
    pca = PCA(nComponents = 2)
    pca.fit(trainNormal)
    trainNormalPCA = pca.transform(trainNormal)
    testNormalPCA = pca.transform(testNormal)
    testAttackPCA = pca.transform(testAttack)

    # Plots PCA results
    plotPCA(trainNormalPCA, testNormalPCA, testAttackPCA)

    # Applies K-Means
    print("\n--- K-Means Results ---")
    kmeans = KMeans(kClusters = 4, maxIters = 100, threshold = 1e-4, initializationAttempts = 10)
    kmeans.fit(trainNormalPCA)

    trainNormalDistribution = kmeans.predict(trainNormalPCA).min(axis = 1)
    testNormalDistribution = kmeans.predict(testNormalPCA).min(axis = 1)
    testAttackDistribution = kmeans.predict(testAttackPCA).min(axis = 1)

    # Thresholding based on training set
    threshold = np.percentile(trainNormalDistribution, 95)

    yTrue = np.array([0] * len(testNormalDistribution) + [1] * len(testAttackDistribution))
    yScores = np.concatenate([testNormalDistribution, testAttackDistribution])
    yPrediction = (yScores > threshold).astype(int)

    confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPrediction)

    print("Confusion Matrix:\n", confusionMatrix)
    print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")

    print("\n--- DBScan Results ---")
    dbscan = DBscan(eps = 0.5, minSamples = 5)
    # Combine normal and attack test sets once
    xTest = np.vstack((testNormalPCA, testAttackPCA))

    # Run DBSCAN and get labels
    dbscan.fit(xTest)
    labels = dbscan.predict(xTest)

    confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPrediction)
    print("Confusion Matrix:\n", confusionMatrix)
    print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")
