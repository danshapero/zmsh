import numpy as np
from fractions import Fraction
import sympy


def _homogeneous(x):
    return (1,) + tuple(Fraction(x_i) for x_i in x)


def _parabolic_lift(x):
    fx = tuple(Fraction(x_i) for x_i in x)
    square_norm = sum(x_i**2 for x_i in fx)
    return (1,) + fx + (square_norm,)


def volume(*args):
    A = sympy.Matrix([_homogeneous(x) for x in args])
    return np.float64(sympy.det(A))


def circumcircle(*args):
    A = sympy.Matrix([_parabolic_lift(x) for x in args])
    return np.float64(sympy.det(A))
