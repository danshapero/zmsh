import numpy as np
from fractions import Fraction


def _determinant(A):
    if A.shape == (2, 2):
        return A[1, 1] * A[0, 0] - A[1, 0] * A[0, 1]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")

    subdets = [
        A[0, k] * _determinant(np.delete(np.delete(A, 0, 0), k, 1))
        for k in range(A.shape[0])
    ]

    return sum(subdets[::2]) - sum(subdets[1::2])


def _homogeneous(x):
    return (1,) + tuple(Fraction(x_i) for x_i in x)


def _parabolic_lift(x):
    fx = tuple(Fraction(x_i) for x_i in x)
    square_norm = sum(x_i**2 for x_i in fx)
    return (1,) + fx + (square_norm,)


def volume(*args):
    A = np.array([_homogeneous(x) for x in args], dtype=object)
    return np.float64(_determinant(A))


def circumcircle(*args):
    A = np.array([_parabolic_lift(x) for x in args], dtype=object)
    return np.float64(_determinant(A))
