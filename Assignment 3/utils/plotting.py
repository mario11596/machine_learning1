import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(X, clf, plot_step=0.05):
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  padding = (x_max - x_min) / 10
  xx, yy = np.meshgrid(np.arange(x_min - padding, x_max + padding, plot_step),
                       np.arange(y_min - padding, y_max + padding, plot_step))

  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)


def plot_dataset(X_train, y_train, X_test=None, y_test=None):
  plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train, label='train', cmap=plt.cm.RdYlBu)
  if X_test is not None and y_test is not None:
    plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=y_test, label='test', cmap=plt.cm.RdYlBu, marker='x')
  plt.legend()
