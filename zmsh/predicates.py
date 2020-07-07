import numpy as np
from fractions import Fraction

def area(x, y, z):
    xq = (Fraction(x[0]), Fraction(x[1]))
    yq = (Fraction(y[0]), Fraction(y[1]))
    zq = (Fraction(z[0]), Fraction(z[1]))

    q = 0.5 * (
        (y[0] * z[1] - y[1] * z[0]) -
        (x[0] * z[1] - x[1] * z[0]) +
        (x[0] * y[1] - x[1] * y[0])
    )

    return np.float64(q)
