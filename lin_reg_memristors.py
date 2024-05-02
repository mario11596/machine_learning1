from enum import Enum
from typing import Tuple
import numpy as np


class MemristorFault(Enum):
    IDEAL = 0
    DISCORDANT = 1
    STUCK = 2
    CONCORDANT = 3


def model_to_use_for_fault_classification():
    return 2


def fit_zero_intercept_lin_model(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """

    # TODO: implement the equation for theta containing sums
    delta_R_ideal = x
    delta_R = y
    numerator = np.sum(delta_R * delta_R_ideal)
    denominator = np.sum(delta_R_ideal ** 2)
    theta = numerator / denominator
    return theta


def bonus_fit_lin_model_with_intercept_using_pinv(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1
    """
    from numpy.linalg import pinv

    # TODO: implement the equation for theta using the pseudo-inverse (Bonus Task)
    theta = [None, None]
    return theta[0], theta[1]


def fit_lin_model_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1 
    """

    # TODO: implement the equation for theta_0 and theta_1 containing sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    n = len(x)

    det_A = n * sum_xx - sum_x * sum_x

    if det_A != 0:
        # Inverse of matrix
        mat_inv = (1 / det_A) * np.array([[sum_xx, -sum_x],
                                        [-sum_x, n]])

        # Multiplying the inverse of A with b
        theta = mat_inv @ np.array([sum_y, sum_xy])
        theta_0, theta_1 = theta[0], theta[1]
    else:
        # Handle the non-invertible case
        print("Matrix A is singular and cannot be inverted.")
        theta_0, theta_1 = None, None

    return theta_0, theta_1


def classify_memristor_fault_with_model1(theta: float) -> MemristorFault:
    """
    :param theta: the estimated parameter of the zero-intercept linear model
    :return: the type of fault
    """
    # TODO: Implement either this function, or the function `classify_memristor_fault_with_model2`,
    #       depending on which model you decide to use.

    # If you decide to use this function, remove the line `raise NotImplementedError()` and
    # return a MemristorFault based on the value of theta.
    # For example, return MemristorFault.IDEAL if you decide that the given theta does not indicate a fault, and so on.
    # Use if-statements and choose thresholds for the parameters that make sense to you.

    raise NotImplementedError()


def classify_memristor_fault_with_model2(theta0: float, theta1: float) -> MemristorFault:
    """
    :param theta0: the intercept parameter of the linear model
    :param theta1: the slope parameter of the linear model
    :return: the type of fault
    """
    margin = 0.5
    if theta1 > 1 + margin:
        fault_type = MemristorFault.CONCORDANT
    elif theta1 < -1 - margin:
        fault_type = MemristorFault.DISCORDANT
    elif -1 + margin < theta1 < 1 - margin:
        fault_type = MemristorFault.STUCK
    else:
        fault_type = MemristorFault.IDEAL

    return fault_type

