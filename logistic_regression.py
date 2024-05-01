import numpy as np

def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 1
    num_rows = X_data.shape[0]
    tmp_array = np.zeros(shape = (num_rows, 1))

    for index in range(num_rows):
        if X_data[index, 0] >= 10 and X_data[index, 1] <= 20:
            tmp_array[index] = 1

    new_feature_x3 = tmp_array
    X = np.append(X_data, new_feature_x3, axis = 1)

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2

    x1_squared = X_data[:, 0] ** 2
    x2_squared = X_data[:, 1] ** 2

    radius = 24
    new_feature_circle = ((x1_squared + x2_squared) - (radius ** 2))

    X = np.append(X_data, new_feature_circle[:, np.newaxis], axis=1)

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    normalize_data = (X_data - np.mean(X_data, axis = 0)) / np.std(X_data, axis = 0)

    new_feature_x1 = normalize_data[:, 0] ** 2
    new_feature_x2 = normalize_data[:, 0] * normalize_data[:, 1]
    new_feature_x3 = normalize_data[:, 1] ** 2

    new_feature_x4 = normalize_data[:, 0] ** 3
    new_feature_x5 = normalize_data[:, 1] ** 3
    new_feature_x6 = new_feature_x3 * new_feature_x1

    X = np.append(normalize_data, new_feature_x1[:, np.newaxis], axis=1)
    X = np.append(X, new_feature_x2[:, np.newaxis], axis=1)
    X = np.append(X, new_feature_x3[:, np.newaxis], axis=1)

    X = np.append(X, new_feature_x4[:, np.newaxis], axis=1)
    X = np.append(X, new_feature_x5[:, np.newaxis], axis=1)
    X = np.append(X, new_feature_x6[:, np.newaxis], axis=1)

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': None}
