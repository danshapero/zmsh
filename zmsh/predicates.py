import numpy as np
from fractions import Fraction
import sympy


def _homogeneous(x):
    return (1,) + tuple(Fraction(x_i) for x_i in x)


def volume(*args):
    A = sympy.Matrix([_homogeneous(x) for x in args])
    return np.float64(sympy.det(A))
