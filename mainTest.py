# python -m pip install --upgrade pip
# python -m pip install numpy matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import os
import itertools

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
    Student-style DBSCAN using a grid-based neighbor search.
    Works in any dimension (PCA=2, PCA=10, etc.).
    Faster than O(n^2) since it only checks nearby cells.
    """

    def __init__(self, eps=0.5, minSamples=5):
        self.eps = eps
        self.minSamples = minSamples
        self.grid = {}       # hash map: cell -> list of (index, point)
        self.corePoints = [] # list of core points
        self.dim = None      # dimension of data

    def _toCell(self, point):
        """
        Convert a point into a grid cell index (tuple).
        Each cell side length = eps.
        """
        return tuple((point // self.eps).astype(int))

    def fit(self, X):
        """
        Build the grid and find core points.
        """
        self.dim = X.shape[1]  # number of dimensions
        self.grid = {}

        # Assign each point to a grid cell
        for i, point in enumerate(X):
            cell = self._toCell(point)
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append((i, point))

        # Identify core points
        self.corePoints = []
        for i, point in enumerate(X):
            if self.countNeighbors(point) >= self.minSamples:
                self.corePoints.append(point)
        self.corePoints = np.array(self.corePoints)

    def countNeighbors(self, point):
        """
        Count neighbors by checking current cell + surrounding cells.
        Dimension-independent.
        """
        cell = self._toCell(point)
        count = 0
        # Generate all neighbor cell offsets: [-1, 0, 1]^dim
        for offset in itertools.product([-1, 0, 1], repeat=self.dim):
            neighborCell = tuple(cell[d] + offset[d] for d in range(self.dim))
            if neighborCell in self.grid:
                for _, otherPoint in self.grid[neighborCell]:
                    # early exit when we've reached minSamples
                    if count >= self.minSamples:
                        return count
                    if np.linalg.norm(point - otherPoint) <= self.eps:
                        count += 1
        return count

    def classifyPoint(self, testPoint):
        """
        Classify a test point as normal (0) if within eps of any core point.
        Otherwise anomaly (1).
        """
        if len(self.corePoints) == 0:
            return 1
        dists = np.linalg.norm(self.corePoints - testPoint, axis=1)
        return 0 if np.any(dists <= self.eps) else 1

    def predict(self, X):
        """
        Predict labels for a set of test points.
        """
        # Vectorized prediction using chunking against corePoints
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
    epsValues = [0.005, 0.01, 0.015, 0.02]
    minSamplesValues = [3, 5, 10]

    for eps in epsValues:
        for minSamples in minSamplesValues:
            dbscan = DBSCAN(eps=eps, minSamples=minSamples)
            dbscan.fit(trainNormalPCA)

            testNormalPred = dbscan.predict(testNormalPCA)
            testAttackPred = dbscan.predict(testAttackPCA)

            yPredDBSCAN = np.concatenate([testNormalPred, testAttackPred])
            yTrue = np.array([0] * len(testNormalPCA) + [1] * len(testAttackPCA))

            confusionMatrix, Accuracy, TPR, FPR, F1 = evaluateResults(yTrue, yPredDBSCAN)
            print(f"\neps = {eps}, Minimum Samples = {minSamples}")
            print("Confusion Matrix:\n", confusionMatrix)
            print(f"Accuracy = {Accuracy:.4f}, TPR = {TPR:.4f}, FPR = {FPR:.4f}, F1 = {F1:.4f}")
