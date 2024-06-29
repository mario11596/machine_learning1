import numpy as np
from sklearn.base import BaseEstimator


def loss(w, b, C, X, y):
    # : Implement the loss function (Equation 1)
    #       useful methods: np.sum, np.clip
    # regularization term
    reg_term = 0.5 * np.dot(w, w)

    # hinge loss using
    hinge_losses = np.clip(1 - y * (np.dot(X, w) + b), 0, None)

    # total loss
    total_loss = reg_term + C * np.sum(hinge_losses)

    return total_loss


def grad(w, b, C, X, y):
    # : Implement the gradients of the loss with respect to w and b.
    #       Useful methods: np.sum, np.where, numpy broadcasting

    # Compute the margins
    margins = y * (np.dot(X, w) + b)

    # Indicator function for hinge loss
    filter = np.where(margins < 1, 1, 0)

    # Gradient with respect to w
    grad_w = w - C * np.sum((filter * y)[:, np.newaxis] * X, axis=0)

    # Gradient with respect to b
    grad_b = -C * np.sum(filter * y)

    return grad_w, grad_b


class LinearSVM(BaseEstimator):
    def __init__(self, C=1, eta=1e-3, max_iter=1000):
        self.b = None
        self.w = None
        self.C = C
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, X, y):
        # convert y such that components are not \in {0, 1}, but \in {-1, 1}
        y = np.where(y == 0, -1, 1)

        # : Initialize self.w and self.b. Does the initialization matter?
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        
        loss_list = []
        eta = self.eta  # starting learning rate
        for j in range(self.max_iter):
            # : Compute the gradients, update self.w and self.b using `eta` as the learning rate.
            #       Compute the loss and add it to loss_list.
            grad_w, grad_b = grad(self.w, self.b, self.C, X, y)

            # Update self.w and self.b using `eta` as the learning rate
            self.w -= eta * grad_w
            self.b -= eta * grad_b

            # Compute the loss and add it to loss_list
            current_loss = loss(self.w, self.b, self.C, X, y)
            loss_list.append(current_loss)

            # decaying learning rate
            eta = eta * 0.99

        return loss_list

    def predict(self, X):
        # : Predict class labels of unseen data points on rows of X
        #       NOTE: The output should be a vector of 0s and 1s (*not* -1s and 1s)
        predictions = np.dot(X, self.w) + self.b
        return np.where(predictions >= 0, 1, 0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
