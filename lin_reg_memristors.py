from enum import Enum
from typing import Tuple
import numpy as np


class MemristorFault(Enum):
    IDEAL = 0
    DISCORDANT = 1
    STUCK = 2
    CONCORDANT = 3


def model_to_use_for_fault_classification():
    return None # TODO: change this to either 1 or 2 (depending on which model you decide to use)


def fit_zero_intercept_lin_model(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """

    # TODO: implement the equation for theta containing sums
    theta = None
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
    theta_0 = None
    theta_1 = None
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
    # TODO: Implement either this function, or the function `classify_memristor_fault_with_model1`,
    #       depending on which model you decide to use.

    # If you decide to use this function, remove the line `raise NotImplementedError()` and
    # return a MemristorFault based on the value of theta0 and theta1.
    # For example, return MemristorFault.IDEAL if you decide that the given theta pair
    # does not indicate a fault, and so on.
    # Use if-statements and choose thresholds for the parameters that make sense to you.

    raise NotImplementedError()
