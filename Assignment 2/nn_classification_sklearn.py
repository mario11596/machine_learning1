from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.

    pca_model = PCA(n_components = n_components)
    x_transform = pca_model.fit_transform(X_train)

    pca_ratio = np.sum(pca_model.explained_variance_ratio_) * 100

    print(f'Explained variance ratio of the PCA object: {pca_ratio}')

    return x_transform, pca_model


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of neurons in one hidden layer.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.

    num_hidden = [2, 10, 100, 200, 500]
    all_models = []

    for n_hidden in num_hidden:
        print(f'The number of neurons in one hidden layer: {n_hidden}')

        mlp_model = MLPClassifier(solver='adam', max_iter=500, random_state=1, hidden_layer_sizes=n_hidden)
        mlp_model.fit(X_train, y_train)

        prediction_train = mlp_model.predict(X_train)
        prediction_validation = mlp_model.predict(X_val)

        count = 0
        for prediction, ground_true in zip(prediction_train, y_train):
            if prediction == ground_true:
                count = count + 1

        accuracy_train = count / len(y_train)

        count = 0
        for prediction, ground_true in zip(prediction_validation, y_val):
            if prediction == ground_true:
                count = count + 1

        accuracy_validation = count / len(y_val)

        all_models.append(mlp_model)

        print(f'Train accuracy: {round(accuracy_train, 5)}')
        print(f'Validation accuracy: {round(accuracy_validation, 5)}')
        print(f'Training loss: {round(mlp_model.loss_, 5)}')


    mlp_model_best = all_models[3]

    return mlp_model_best


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.
    mlp_model = None
    num_hidden = [2, 10, 100, 200, 500]
    regularization = [[0.1, None],
                      [None, True],
                      [0.1, True]]
    all_models = []

    for n_hidden in num_hidden:
        for i in range(3):
            if i == 0:
                print(f'The number of neurons in one hidden layer: {n_hidden}, alpha: {regularization[i][0]}')
                mlp_model = MLPClassifier(solver='adam', max_iter=500, random_state=1, hidden_layer_sizes=n_hidden, alpha=regularization[i][0])
            elif i == 1:
                print(f'The number of neurons in one hidden layer: {n_hidden}, early_stopping: {regularization[i][1]}')
                mlp_model = MLPClassifier(solver='adam', max_iter=500, random_state=1, hidden_layer_sizes=n_hidden,early_stopping=regularization[i][1])
            elif i == 2:
                print(f'The number of neurons in one hidden layer: {n_hidden}, alpha: {regularization[i][0]}, early_stopping: {regularization[i][1]}')
                mlp_model = MLPClassifier(solver='adam', max_iter=500, random_state=1, hidden_layer_sizes=n_hidden,
                                          early_stopping=regularization[i][1], alpha=regularization[i][0])
            mlp_model.fit(X_train, y_train)

            prediction_train = mlp_model.predict(X_train)
            prediction_validation = mlp_model.predict(X_val)

            count = 0
            for prediction, ground_true in zip(prediction_train, y_train):
                if prediction == ground_true:
                    count = count + 1

            accuracy_train = count / len(y_train)

            count = 0
            for prediction, ground_true in zip(prediction_validation, y_val):
                if prediction == ground_true:
                    count = count + 1

            accuracy_validation = count / len(y_val)

            all_models.append(mlp_model)

            print(f'Train accuracy: {round(accuracy_train, 5)}')
            print(f'Validation accuracy: {round(accuracy_validation, 5)}')
            print(f'Training loss: {round(mlp_model.loss_, 5)}')


    return all_models[12]


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.

    plt.plot(nn.loss_curve_)
    plt.ylabel('Loss value')
    plt.xlabel('Iteration')
    plt.title('Training loss curve for the MLPClassifier')
    plt.grid(True)
    plt.savefig('training_loss_mlpcclassifier.png')
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.

    model_prediction = nn.predict(X_test)

    test_accuracy = nn.score(X_test, y_test)
    print(f'Test accuracy: {round(test_accuracy, 5)}')

    cf_matrix = confusion_matrix(y_test, model_prediction)
    cd_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=nn.classes_)
    cd_display.plot()
    plt.savefig('confusion_matrix.png')
    plt.show()

    cd_report = classification_report(y_test, model_prediction)
    print(cd_report)

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    parameters = {
        'alpha' : [0.0, 0.1, 1.0],
        'solver': ['lbfgs', 'adam'],
        'hidden_layer_sizes' : [(100,), (200)]
    }

    mlp_model = MLPClassifier(max_iter=100, random_state=42)
    grid_search = GridSearchCV(mlp_model, parameters, cv = 5, verbose = 4)
    grid_search.fit(X_train, y_train)

    print(f'Results after Grid serach. The best score: {grid_search.best_score_}, the best parameter set: {grid_search.best_params_}')
    return grid_search.best_estimator_