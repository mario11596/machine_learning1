from sklearn.model_selection import train_test_split
from mlp_classifier_own import MLPClassifierOwn
import numpy as np


def train_nn_own(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifierOwn:
    """
    Train MLPClassifierOwn with PCA-projected features.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifierOwn object
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Create a MLPClassifierOwn object and fit it using (X_train, y_train)
    #       Print the train accuracy and validation accuracy
    #       Return the trained model

    # mlp_class = MLPClassifierOwn(5, 0.0, 32, (16,), 42)
    # mlp_class.fit(X_train, y_train)
    # print("alpha = 0.0")
    # print("-----------------------------------------")
    # mlp_class = MLPClassifierOwn(5, 1.0, 32, (16,), 42)
    # mlp_class.fit(X_train, y_train)
    # print("alpha = 1.0")
    # print("-----------------------------------------")
    mlp_class = MLPClassifierOwn(5, 2.0, 32, (16,), 42)
    mlp_class.fit(X_train, y_train)
    print("alpha = 2.0")
    print("-----------------------------------------")
    # mlp_class = MLPClassifierOwn(5, 10.0, 32, (16,), 42)
    # mlp_class.fit(X_train, y_train)
    # print("alpha = 10.0")
    # print("-----------------------------------------")
    return mlp_class
