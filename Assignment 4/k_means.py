from typing import Tuple
import numpy as np


def wcss(X: np.ndarray, K: int, Z: np.ndarray, centroids: np.ndarray) -> float:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension of data points
    :param K: number of clusters
    :param Z: indicator variables for all data points, shape: (N, K)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: objective function WCSS - a scalar value
    """

    # TODO: Calculate WCSS and return it

    wcss = 0
    for data in range(X.shape[0]):
        for cluster_k in range(K):
            if Z[data, cluster_k] == 1:
                wcss += np.sum((X[data] - centroids[cluster_k]) ** 2)

    return wcss


def closest_centroid(sample: np.ndarray, centroids: np.ndarray) -> int:
    """
    :param sample: a data point x_n (of dimension D)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: idx_closest_cluster, that is, the index of the closest cluster
    """

    # TODO: Calculate distance of the current sample to each centroid.
    #       Afterwards you should return the index of the closest centroid (int value from 0 to (K-1))
    distance_euclidian = np.sqrt(np.sum((sample - centroids) ** 2, axis=1))
    min_distance = np.argmin(distance_euclidian)

    return min_distance


def compute_Z(X: np.ndarray, K: int, centroids: np.ndarray) -> np.ndarray:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: Z: indicator variables for all data points, shape: (N, K)
    """

    N = X.shape[0]
    # TODO: Compute Z matrix which holds the indicator variables for all data points (using `closest_centroid`).
    #       The indicator variables represent the cluster assignments of each data point.
    Z = np.zeros((N, K))

    for i_data in range(N):
        sample_data = X[i_data]

        assign_point_to_cluster = closest_centroid(sample_data, centroids)
        Z[i_data, assign_point_to_cluster] = 1

    assert len(np.unique(Z)) == 2 and np.min(Z) == 0 and np.max(Z) == 1, 'Z should be a matrix of zeros and ones'
    assert np.all(np.sum(Z, axis=1) == np.ones(Z.shape[0])), 'Each data point should be assigned to exactly 1 cluster'
    return Z


def recompute_centroids(X: np.ndarray, K: int, Z: np.ndarray) -> np.ndarray:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param Z: indicator variables for all data points, shape: (N, K)
    :return: centroids - means of clusters, shape: (K, D)
    """

    D = X.shape[1]
    # TODO: Recompute centroids
    centroids = np.zeros((K, D))

    for k in range(K):
        count_data_cluster = 0

        for i in range(X.shape[0]):
            if Z[i, k] == 1:
                centroids[k] += X[i]
                count_data_cluster += 1

        if count_data_cluster > 0:
            current_mean = centroids[k]
            new_mean = current_mean / count_data_cluster
            centroids[k] = new_mean

    return centroids


def kmeans(X: np.ndarray, K: int, max_iter: int, eps=1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter: maximum number of iterations for the K-means algorithm.
                     If the algorithm converges earlier, it should stop.
    :return: Z - indicator variables for all data points, shape: (N, K)
             centroids - means of clusters, shape: (K, D)
             wcss_list - list with values of the objective function J over iteration
    """

    N, D = X.shape

    # Init centroids
    rnd_points = np.random.choice(np.arange(N), size=K, replace=False)
    centroids = X[rnd_points, :]
    assert centroids.shape[0] == K and centroids.shape[1] == D
    print(f'Init: {centroids=}')

    wcss_list = []
    for it in range(max_iter):
        # Assign samples to the clusters (compute Z)
        Z = compute_Z(X, K, centroids) # TODO: function call to assign samples to clusters
        loss = wcss(X, K, Z, centroids) # TODO: function call to calculate WCSS
        wcss_list.append(loss)

        # Calculate new centroids from the clusters
        centroids = recompute_centroids(X, K, Z) # TODO: function call to recompute centroids
        loss = wcss(X, K, Z, centroids) # TODO: function call to calculate WCSS (again)
        wcss_list.append(loss)

        if it > 0 and np.abs(wcss_list[-1] - wcss_list[-2]) < eps:
            print(f'Algorithm converged at iteration {it}.')
            break

    print(f'Fitted parameters: {centroids=}')
    return Z, centroids, np.array(wcss_list)
