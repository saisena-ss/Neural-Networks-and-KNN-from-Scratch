# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self.X_train = X
        self.y_train = y
        self.unique_class = np.unique(y)
        # raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        predictions = []
        for row in X:
            predictions.append(self._predict(row))
        
        return np.array(predictions)
        

    def _predict(self,x):
        distances = np.array([self._distance(row,x) for row in self.X_train])
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        y_labels = list(self.y_train[nearest_indices])
        nearest_dist = distances[nearest_indices] + 0.0001 #adding small number to all to avoid zeros

        final = []
        if self.weights == 'uniform':
            for label in self.unique_class:
                final.append((label,y_labels.count(label)))
            # print(final)
            # final.sort(key = lambda x:x[1],reverse = True)
            return max(final,key = lambda x:x[1])[0]
        
        elif self.weights == 'distance':
            final = {}
            for label_idx in range(len(y_labels)):
                final[y_labels[label_idx]] = final.get(y_labels[label_idx],0) + (1/nearest_dist[label_idx]) # if nearest_dist[label_idx]!=0 else 10*-1000)
            return max(final,key = final.get)
        # raise NotImplementedError('This function must be implemented by the student.')
