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
            distance_tmp = np.sqrt(np.sum((x_test - self.X) ** 2, axis=1))
            knn_index_sort = np.argsort(distance_tmp)
            first_k = knn_index_sort[:self.k]
            knn_class_labels = self.y[first_k]

            count_zero = np.sum(knn_class_labels == 0)
            count_one = np.sum(knn_class_labels == 1)

            if count_zero > count_one:
                decision = 0
            else:
                decision = 1
            prediction_class.append(decision)

        array_prediction_class = np.array(prediction_class)
        return array_prediction_class

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
