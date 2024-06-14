import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from utils.plotting import plot_dataset, plot_decision_boundary
from utils.datasets import get_toy_dataset
from knn import KNearestNeighborsClassifier
from svm import LinearSVM
from sklearn.ensemble import RandomForestClassifier


def grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name):
    knn = KNearestNeighborsClassifier()
    # TODO: Use the `GridSearchCV` meta-classifier and search over different values of `k`
    #       Include the `return_train_score=True` option to get the training accuracies
    k = list(range(1, 101))
    parameters = {
        'k': k
    }
    grid_search = GridSearchCV(knn, parameters, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)

    # this plots the decision boundary
    plt.figure()
    plot_decision_boundary(X_train, grid_search)
    plot_dataset(X_train, y_train, X_test, y_test)
    plt.title(f"Decision boundary for dataset {dataset_name}\nwith k={grid_search.best_params_['k']}")
    # TODO you should use the plt.savefig(...) function to store your plots before calling plt.show()
    plt.savefig(f'knn_with_decision_boundary_dataset_{dataset_name}.png')
    plt.show()

    # TODO: Create a plot that shows the mean training and validation scores (y axis)
    #       for each k \in {1,...,100} (x axis).
    #       Hint: Check the `cv_results_` attribute of the `GridSearchCV` object

    plt.figure()
    plt.xlabel('k value')
    plt.ylabel('Mean score')
    plt.title(f"Mean training and validation scores - dataset {dataset_name}")
    plt.plot(k, grid_search.cv_results_['mean_train_score'], label='Mean train score', color='red')
    plt.plot(k, grid_search.cv_results_['mean_test_score'], label='Mean validation score', color='blue')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'knn_mean_training_validation_dataset_{dataset_name}.png')
    plt.show()


def task1_2():
    print('-' * 10, 'Task 1.2', '-' * 10)
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name=idx)


def task1_4():
    print('-' * 10, 'Task 1.4', '-' * 10)
    dataset_name = '2 (noisy)'
    X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
    for k in [1, 30, 100]:
        # TODO: Fit your KNearestNeighborsClassifier with k in {1, 30, 100} and plot the decision boundaries.
        #       You can use the `cross_val_score` method to manually perform cross-validation.
        #       Report the mean cross-validated scores.
        knn = KNearestNeighborsClassifier(k=k)
        knn.fit(X_train, y_train)

        cv_score = cross_val_score(knn, X_train, y_train)
        mean_scores = np.mean(cv_score)
        print(f"Knn mean cross-validation score is: k={k} {mean_scores}")

        # This plots the decision boundaries without the test set
        # (we simply don't pass the test sets to `plot_dataset`).
        plt.figure()
        plt.title(f"Decision boundary for dataset {dataset_name}\nwith k={k}")
        plot_decision_boundary(X_train, knn)
        plot_dataset(X_train, y_train)
        plt.show()

    # This should find the best parameters for the noisy dataset.
    grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name=dataset_name)


def task2_2():
    print('-' * 10, 'Task 2.2', '-' * 10)

    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    svm = LinearSVM()
    # : Use grid search to find suitable parameters.
    param_grid = {'C': [0.1, 1, 10, 50, 100], 'eta': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # : Use the parameters you have found to instantiate a LinearSVM.
    #       The `fit` method returns a list of scores that you should plot in order to monitor the convergence.
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)
    svm = LinearSVM(C=best_params['C'], eta=best_params['eta'])
    loss_list = svm.fit(X_train, y_train)

    # This plots the decision boundary
    plt.figure()
    plot_dataset(X_train, y_train, X_test, y_test)
    plot_decision_boundary(X_train, svm)
    plt.show()


def task2_3():
    print('-' * 10, 'Task 2.3', '-' * 10)
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        svc = SVC(tol=1e-4)
        # : Perform grid search, decide on suitable parameter ranges
        #       and state sensible parameter ranges in your report
        param_grid = [
            {'C': [0.1, 1, 10, 50, 100], 'kernel': ['linear']},
            {'C': [0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10, 15, 25]}
        ]

        # Create a grid search object with cross-validation
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # : Using the best parameter settings, report the score on the test dataset (X_test, y_test)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        test_score = grid_search.score(X_test, y_test)

        print(f"Dataset {idx}>> Best params: {best_params}")
        print(f"Dataset {idx}>> Mean cv acc: {best_score:.4f}")
        print(f"Dataset {idx}>> Test acc: {test_score:.4f}")
        print("=============================================================\n")

        # This plots the decision boundary
        plt.figure()
        plt.title(f"[SVM] Decision boundary for dataset {idx}\nwith params={grid_search.best_params_}")
        plot_dataset(X_train, y_train, X_test, y_test)
        plot_decision_boundary(X_train, grid_search)
        plt.show()


def task3_1():
    n_estimators_list = [1, 100]
    max_depth_list = np.arange(1, 26)
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        cv_val_accuracy = {}
        cv_train_accuracy = {}
        for n_estimators in n_estimators_list:
            # TODO: Instantiate a RandomForestClassifier with n_estimators and random_state=0
            #       and use GridSearchCV over max_depth_list to find the best max_depth.
            #       Use `return_train_score=True` to get the training accuracies during CV.
            random_forest_clss = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

            parameters = {
                'max_depth': max_depth_list
            }

            grid_search = GridSearchCV(estimator=random_forest_clss, param_grid=parameters, cv=5, return_train_score=True)
            grid_search.fit(X_train, y_train)

            print(f'The best parameters are: {grid_search.best_params_}')

            # TODO: Store `mean_test_score` and `mean_train_score` in cv_val_accuracy and cv_train_accuracy.
            #       The dictionary key should be the number of estimators.
            #       Hint: Check the `cv_results_` attribute of the `GridSearchCV` object
            cv_val_accuracy[n_estimators] = grid_search.cv_results_['mean_test_score']
            cv_train_accuracy[n_estimators] = grid_search.cv_results_['mean_train_score']

            cv_train = grid_search.cv_results_['mean_train_score'].mean()
            cv_test = grid_search.cv_results_['mean_test_score'].mean()
            print(f'The mean cross-validation accuracy for validation with n_estimator {n_estimators} is: {cv_test}')
            print(f'The mean cross-validation accuracy for training with n_estimator {n_estimators} is: {cv_train}')

            # This plots the decision boundary with just the training dataset
            plt.figure()
            plot_decision_boundary(X_train, grid_search)
            plot_dataset(X_train, y_train)
            plt.title(f"Decision boundary for dataset {idx}\n"
                      f"n_estimators={n_estimators}, max_depth={grid_search.best_params_['max_depth']}")
            plt.show()

        # TODO: Create a plot that shows the mean training and validation scores (y axis)
        #       for each max_depth in max_depth_list (x axis).
        #       Use different colors for each n_estimators and linestyle="--" for validation scores.
        color_estimator = {
            1: 'red',
            100: 'blue'
        }
        plt.plot(max_depth_list, cv_train_accuracy[1], color=color_estimator[1], label='Train with estimator 1')
        plt.plot(max_depth_list, cv_train_accuracy[100], color=color_estimator[100], label='Train with estimator 100')

        plt.plot(max_depth_list, cv_val_accuracy[1], color=color_estimator[1], linestyle="--", label='Validation with estimator 1')
        plt.plot(max_depth_list, cv_val_accuracy[100], color=color_estimator[100],linestyle="--", label='Validation with estimator 100')

        plt.legend()
        plt.ylabel('Mean values')
        plt.xlabel('Max depth')
        plt.grid(True)
        plt.savefig(f'mean_values_with_estimator_1_and_100_dataset_{idx}')
        # TODO: Instantiate a RandomForestClassifier with the best parameters for each dataset and
        #       report the test scores (using X_test, y_test) for each dataset.

        classifier = RandomForestClassifier(n_estimators=100, max_depth=grid_search.best_params_['max_depth'])
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Test score
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print(f"[TEST_EVAL] Dataset {idx}, accuracy: {accuracy}, best_params: n_estimators=100, {grid_search.best_params_}")


def task3_bonus():
    X_train, X_test, y_train, y_test = get_toy_dataset(4)

    # TODO: Find suitable parameters for an SVC and fit it.
    #       Report mean CV accuracy of the model you choose.

    # TODO: Fit a RandomForestClassifier with appropriate parameters.

    # TODO: Create a `barh` plot of the `feature_importances_` of the RF classifier.
    #       See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

    # TODO: Use recursive feature elimination to automatically choose the best number of parameters.
    #       Set `scoring='accuracy'` to look for the feature subset with highest accuracy and fit the RFECV
    #       to the training dataset. You can pass the classifier from the previous step to RFECV.
    #       See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

    # TODO: Use the RFECV to transform the training dataset -- it automatically removes the least important
    #       feature columns from the datasets. You don't have to change y_train or y_test.
    #       Fit an SVC classifier with appropriate parameters on the new dataset and report the mean CV score.
    #       Do you see a difference in performance when compared to the previous dataset? Report your findings.

    # TODO: If the CV performance of this SVC is better, transform the test dataset as well and report the test score.
    #       If the performance is worse, report the test score of the previous SVC.


if __name__ == '__main__':
    # Task 1.1 consists of implementing the KNearestNeighborsClassifier class
    # task1_2()
    # Task 1.3 does not need code to be answered
    # task1_4()

    # Task 2.1 consists of a pen & paper exercise and the implementation of the LinearSVM class
    # task2_2()
    # task2_3()

    task3_1()
    # Task 3.2 is a theory question
    # task3_bonus()
