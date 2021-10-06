"""
This is a homemade implementing of K Means Clustering from scratch in Python.

The steps of K Means Clustering can be summarized:

Step 1: randomly choose k center points (centroids)

Step 2 (assignment step): for each observation, assign it to the cluster with the least Euclidean distance

Step 3 (update step): re-calculate means (centroids) for observation assigned to each cluster

Step 4: stop when the assignment no longer change
"""

import numpy as np

class KMeansCluster():

    def __init__(self, k=1) -> None:
        """ 
        Initialize 

        Parameters
        ----------
        k : the number of clusters want to generate

        """
        self.k = k
        self.X = None
        self.centroids = None
        self.labels = []

    def _distance(self, x1, x2):
        """
        Return the Euclidean distance between two data points 
        """
        assert x1.shape == x2.shape
        
        return np.sqrt(np.sum(np.power(x1 - x2, 2)))

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : d-dimentional numpy array where each row is an observation, each column is a dimension

        Returns:
        self : fitted estimator
        """
        self.X = X

        # step 1: randomly choose the index of k centroids
        nrows, ncols = X.shape
        index_centroids = np.random.choice(nrows, size=self.k, replace=False)
        centroids = [X[i] for i in index_centroids]
        labels = [0]*nrows
        # store last group of labels
        self.labels = labels

        while True:
            # step 2 (assignment step): 
            # loop through each observation
            for i in range(nrows):
                # calculate the distance to each centroids
                label = 0
                dist = self._distance(X[i], centroids[0])
                for j in range(1, self.k):
                    # update label and dist if find a smaller distance
                    if self._distance(X[i], centroids[j]) < dist:
                        dist  = self._distance(X[i], centroids[j])
                        label = j

                # update label
                labels[i] = label

            # step 3 (update step):
            sums = np.zeros((self.k, ncols))
            nums = [0]*self.k
            for i in range(nrows):
                # self.labels stored labels for each row
                sums[labels[i]] += X[i]
                nums[labels[i]] += 1
            
            # update centroids by averaging
            for j in range(self.k):
                centroids[j] = sums[j] / nums[j]
                
            # step 4: check if the stopping condition is satisfied
            if self.labels == labels:
                self.centroids = centroids
                break


    def predict(self, data):
        """
        Return a list of labels that a lable is the predicted cluster to a observation in data
        """
        nrows, ncols = data.shape

        labels = [0]*nrows
        for i in range(nrows):
            # calculate the distance to each centroids
            label = 0
            dist = self._distance(data[i], self.centroids[0])
            for j in range(1, self.k):
                # update label and dist if find a smaller distance
                if self._distance(data[i], self.centroids[j]) < dist:
                    dist  = self._distance(data[i], self.centroids[j])
                    label = j
            # update label
            labels[i] = label

        return labels
        