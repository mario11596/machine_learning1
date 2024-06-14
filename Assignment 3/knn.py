import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None
        self.classes = None # a list of unique classes in our classification problem

    def fit(self, X, y):
        # TODO: Implement this method by storing X, y and infer the unique classes from y
        #       Useful numpy methods: np.unique
        self.y = y
        self.X = X
        self.classes = np.unique(y)

        return self

    def predict(self, X):
        # TODO: Predict the class labels for the data on the rows of X
        #       Useful numpy methods: np.argsort, np.argmax
        #       Broadcasting is really useful for this task.
        #       See https://numpy.org/doc/stable/user/basics.broadcasting.html

        prediction_class = []

        for x_test in X:
            distance_tmp = np.sqrt(np.sum((x_test - self.X) **2, axis=1))
            knn_index = np.argsort(distance_tmp)[:self.k]
            knn_y_labels = self.y[knn_index]

            num_zero = np.sum(knn_y_labels == 0)
            num_one = np.sum(knn_y_labels == 1)

            if num_zero > num_one:
                decision = 0
            else:
                decision = 1
            prediction_class.append(decision)

        return np.array(prediction_class)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
