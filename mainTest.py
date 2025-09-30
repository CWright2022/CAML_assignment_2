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
                # Assign clusters using vectorized distances (faster than Python loops)
                # distances shape: (n_samples, n_centroids)
                distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
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

    def distanceToCentroids(self, X):
        """
        Return the minimum distance to a centroid for each sample.
        Useful for anomaly detection thresholding.
        """
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.min(distances, axis=1)
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis = 1)

class DBSCAN:
    """
    Simplified DBSCAN for anomaly detection.
    Finds core points from training data and classifies
    test points as normal (0) or anomaly (1).
    """

    def __init__(self, eps=0.5, minSamples=5):
        self.eps = eps
        self.minSamples = minSamples
        self.corePoints = None
        self.points = None

    def fit(self, X):
        self.points = X
        coreList = []
        for i, point in enumerate(X):
            distances = np.linalg.norm(X - point, axis=1)
            neighbors = np.where(distances <= self.eps)[0]
            if len(neighbors) >= self.minSamples:
                coreList.append(point)
        self.corePoints = np.array(coreList)

    def classifyPoint(self, testPoint):
        if self.corePoints is None or len(self.corePoints) == 0:
            return 1  # if no core points, everything is anomaly
        dists = np.linalg.norm(self.corePoints - testPoint, axis=1)
        if np.any(dists <= self.eps):
            return 0
        return 1

    def predict(self, X):
        return np.array([self.classifyPoint(p) for p in X])


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

    # Use distance-to-nearest-centroid as anomaly score
    trainDistances = kmeans.distanceToCentroids(trainNormalPCA)
    testNormalDistances = kmeans.distanceToCentroids(testNormalPCA)
    testAttackDistances = kmeans.distanceToCentroids(testAttackPCA)

    # Thresholding based on training distances (95th percentile)
    threshold = np.percentile(trainDistances, 95)

    yTrue = np.array([0] * len(testNormalDistances) + [1] * len(testAttackDistances))
    yScores = np.concatenate([testNormalDistances, testAttackDistances])
    yPrediction = (yScores > threshold).astype(int)

    confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPrediction)

    print("Confusion Matrix:\n", confusionMatrix)
    print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")

    print("\n--- DBSCAN Results ---")
    dbscan = DBSCAN(eps=0.5, minSamples=5)
    dbscan.fit(trainNormalPCA)  # learn core points from training normal

    testNormalPred = dbscan.predict(testNormalPCA)
    testAttackPred = dbscan.predict(testAttackPCA)

    yPredDBSCAN = np.concatenate([testNormalPred, testAttackPred])
    yTrue = np.array([0] * len(testNormalPCA) + [1] * len(testAttackPCA))

    confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPredDBSCAN)
    print("Confusion Matrix:\n", confusionMatrix)
    print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")
