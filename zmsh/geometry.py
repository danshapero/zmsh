import itertools
import numpy as np
from .topology import Topology


def compositions(number, size):
    r"""Compute the number of weak compositions of `number` of `size` integers

    From this [SO answer](https://stackoverflow.com/a/40540014)
    """
    m = number + size - 1
    last = (m,)
    first = (-1,)
    for t in itertools.combinations(range(m), size - 1):
        yield tuple(v - u - 1 for u, v in zip(first + t, t + last))


class Geometry:
    def __init__(self, topology: Topology, points: np.ndarray):
        r"""A piecewise linear complex with the given underlying topology"""
        self._topology = topology
        self._points = points

    @property
    def topology(self):
        return self._topology

    @property
    def points(self):
        return self._points

    @property
    def dimension(self):
        return self.points.shape[1]
