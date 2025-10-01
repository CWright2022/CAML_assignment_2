# Required Libraries
# pip install numpy matplotlib if not already installed
import matplotlib.pyplot as plt 
import numpy as np
import os
import itertools

# Helper Functions
def euclideanDistance(pointOne, pointTwo):
    """
    Finds the Euclidean distance (straight-line) between two points.
    This is the main metric used for clustering.
    """
    eDistance = float(np.linalg.norm(pointOne - pointTwo))
    return eDistance

def loadDataset():
    """
    Loads the training, testing normal, and testing attack datasets.
    Data is stored as .npy files in the dataset folder.
    """
    dataDirectory = "./dataset"
    trainNormal = np.load(os.path.join(dataDirectory, "training_normal.npy")).astype(np.float64)
    testNormal = np.load(os.path.join(dataDirectory, "testing_normal.npy")).astype(np.float64)
    testAttack = np.load(os.path.join(dataDirectory, "testing_attack.npy")).astype(np.float64)
    return trainNormal, testNormal, testAttack

def summarizeData(dataName, array):
    """
    Prints a quick summary (shape, mean, std dev) of the dataset.
    Helps us understand if the data looks balanced or has outliers.
    """
    print(f"Summary of {dataName}:")
    print(f"Shape: {array.shape}")
    print(f"Mean: {np.mean(array):.4f}")
    print(f"Standard Deviation: {np.std(array):.4f}")

def evaluateResults(yTrue, yPred):
    """
    Builds a confusion matrix and calculates Accuracy, TPR, FPR, F1.
    These metrics help  evaluate anomaly detection performance.
    """
    TN, FP, FN, TP = np.bincount(2 * yTrue + yPred, minlength = 4)
    confusionMatrix = np.array([[TN, FP], [FN, TP]])

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return confusionMatrix, Accuracy, TPR, FPR, F1

# PCA Implementation
class PCA:
    """
    Principal Component Analysis from scratch.
    Uses covariance matrix + eigen decomposition.
    """
    def __init__(self, nComponents):
        self.nComponents = nComponents
        self.mean = None
        self.components = None

    def fit(self, X):
        # Centers the dataset
        self.mean = np.mean(X, axis = 0)
        xCentered = X - self.mean

        # Covariance matrix
        covarianceMatrix = np.cov(xCentered, rowvar = False)

        # Eigen decomposition
        eigenValues, eigenVectors = np.linalg.eigh(covarianceMatrix)

        # Sorts eigenvalues in descending order
        sortedIndices = np.argsort(eigenValues)[::-1]
        self.components = eigenVectors[:, sortedIndices[:self.nComponents]]

    def transform(self, X):
        # Projects data onto principal components
        xCentered = X - self.mean
        return np.dot(xCentered, self.components)
    
def plotPCA(trainNormalPCA, testNormalPCA, testAttackPCA):
    """
    Visualizes PCA results but only works if 2 components are used.
    Helps see how normal vs. attack traffic separate.
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

# K-Means Implementation
class KMeans:
    """
    K-Means clustering.
    Runs multiple random initializations and keeps the best result.
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

        # Attempts multiple initializations
        for attempt in range(self.initAttempts):
            # Random starting centroids
            randomIndices = np.random.choice(X.shape[0], self.kClusters, replace = False)
            centroids = X[randomIndices]

            # Repeats until convergence is hit
            for _ in range(self.maxIters):
                
                # Assign clusters by closest centroid where they are vectorized for speed
                distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
                labels = np.argmin(distances, axis=1)

                # Update centroids
                newCentroids = []
                for k in range(self.kClusters):
                    if np.any(labels == k):
                        newCentroids.append(X[labels == k].mean(axis=0))
                    else:
                        # If a cluster is empty, code reassigns randomly
                        newCentroids.append(X[np.random.randint(0, X.shape[0])])
                newCentroids = np.array(newCentroids)

                # Check convergence
                if np.linalg.norm(newCentroids - centroids) < self.threshold:
                    break
                centroids = newCentroids

            # Calculates inertia where a lower value is equivalent to better clustering
            inertia = np.sum(np.min(distances, axis=1) ** 2)

            if inertia < bestInertia:
                bestInertia = inertia
                bestCentroids = centroids

        self.centroids = bestCentroids

    def distanceToCentroids(self, X):
        """
        Returns distance to nearest centroid for each point.
        Useful for anomaly detection threshold.
        """
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.min(distances, axis=1)
    
    def predict(self, X):
        """
        Assigns each point to the closest centroid.
        """
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis = 1)

# DBSCAN Class Implementation
class DBSCAN:
    """
    Student-style DBSCAN using grid-based neighbor search.
    Identifies dense clusters and flags outliers.
    """

    def __init__(self, eps=0.5, minSamples=5):
        self.eps = eps
        self.minSamples = minSamples
        self.grid = {}       # Cell to points
        self.corePoints = [] # Core points
        self.dim = None      # Dataset dimension

    def _toCell(self, point):
        """
        Assigns a point to a grid cell.
        Each cell size equals eps.
        """
        return tuple((point // self.eps).astype(int))

    def fit(self, X):
        """
        Builds a grid and identifies core points.
        """
        self.dim = X.shape[1]
        self.grid = {}

        # Assigns points to cells
        for i, point in enumerate(X):
            cell = self._toCell(point)
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append((i, point))

        # Identifies any core points
        self.corePoints = []
        for i, point in enumerate(X):
            if self.countNeighbors(point) >= self.minSamples:
                self.corePoints.append(point)
        self.corePoints = np.array(self.corePoints)

    def countNeighbors(self, point):
        """
        Counts neighbors by scanning current and surrounding cells.
        """
        cell = self._toCell(point)
        count = 0
        for offset in itertools.product([-1, 0, 1], repeat=self.dim):
            neighborCell = tuple(cell[d] + offset[d] for d in range(self.dim))
            if neighborCell in self.grid:
                for _, otherPoint in self.grid[neighborCell]:
                    if count >= self.minSamples:
                        return count
                    if np.linalg.norm(point - otherPoint) <= self.eps:
                        count += 1
        return count

    def classifyPoint(self, testPoint):
        """
        Classifies a point as the following:
        0 = normal (close to cluster core)
        1 = anomaly (too far from any core)
        """
        if len(self.corePoints) == 0:
            return 1
        dists = np.linalg.norm(self.corePoints - testPoint, axis=1)
        return 0 if np.any(dists <= self.eps) else 1

    def predict(self, X):
        """
        Predicts labels for all test points.
        Utilizes chunking to avoid memory blowup.
        """
        if len(self.corePoints) == 0:
            return np.ones(X.shape[0], dtype=int)

        nTest = X.shape[0]
        nCore = self.corePoints.shape[0]
        maxBytes = 200 * 1024 ** 2
        bytesEntry = 8
        blockSize = max(1, int(maxBytes / (nCore * bytesEntry)))
        blockSize = min(max(1, blockSize), 5000)

        predictions = np.ones(nTest, dtype = int)
        for start in range(0, nTest, blockSize):
            end = min(nTest, start + blockSize)
            block = X[start:end]
            dists = np.linalg.norm(block[:, None, :] - self.corePoints[None, :, :], axis=2)
            within = np.any(dists <= self.eps, axis = 1)
            predictions[start:end][within] = 0
        return predictions

# Main Execution
if __name__ == "__main__":
    # Load the data
    trainNormal, testNormal, testAttack = loadDataset()

    # Provides the data summaries
    summarizeData("Training Normal", trainNormal)
    summarizeData("Testing Normal", testNormal)
    summarizeData("Testing Attack", testAttack)

    # PCA reduction
    pca = PCA(nComponents = 2)
    pca.fit(trainNormal)
    trainNormalPCA = pca.transform(trainNormal)
    testNormalPCA = pca.transform(testNormal)
    testAttackPCA = pca.transform(testAttack)

    # Plots the PCA
    plotPCA(trainNormalPCA, testNormalPCA, testAttackPCA)

    # K-Means  Execution
    print("\nK-Means Results:\n")
    kmeans = KMeans(kClusters = 4, maxIters = 100, threshold = 1e-4, initializationAttempts = 10)
    kmeans.fit(trainNormalPCA)

    # Distance-based anomaly scoring
    trainDistances = kmeans.distanceToCentroids(trainNormalPCA)
    testNormalDistances = kmeans.distanceToCentroids(testNormalPCA)
    testAttackDistances = kmeans.distanceToCentroids(testAttackPCA)

    # 95th percentile threshold from training set
    threshold = np.percentile(trainDistances, 95)

    # BuildsS labels
    yTrue = np.array([0] * len(testNormalDistances) + [1] * len(testAttackDistances))
    yScores = np.concatenate([testNormalDistances, testAttackDistances])
    yPrediction = (yScores > threshold).astype(int)

    # Evaluate Metrics for K-Means
    confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPrediction)
    print("Confusion Matrix:\n", confusionMatrix)
    print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")

    # DBSCAN Execution
    print("\nDBSCAN Results:")
    epsValues = [0.005, 0.01, 0.015, 0.02]
    minSamplesValues = [3, 5, 10]

    for eps in epsValues:
        for minSamples in minSamplesValues:
            dbscan = DBSCAN(eps = eps, minSamples = minSamples)
            dbscan.fit(trainNormalPCA)

            testNormalPrediction = dbscan.predict(testNormalPCA)
            testAttackPrediction = dbscan.predict(testAttackPCA)

            yPredictionDBSCAN = np.concatenate([testNormalPrediction, testAttackPrediction])
            yTrue = np.array([0] * len(testNormalPCA) + [1] * len(testAttackPCA))

            # Evaluate Metrics for DBSCAN
            confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPredictionDBSCAN)
            print(f"\neps = {eps}, Minimum Samples = {minSamples}")
            print("Confusion Matrix:\n", confusionMatrix)
            print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")
