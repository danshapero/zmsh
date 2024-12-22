import numpy as np
from . import simplicial, polytopal


def compute_plane(xs: np.ndarray) -> np.ndarray:
    # NOTE: We need to be *very* careful about the shape of the `xs` array --
    # is each point in a row or a column? Here I'm assuming it's by rows
    A = np.column_stack((np.ones(xs.shape[0]), xs))
    z = np.array([np.det(np.delete(A, k, axis=1)) for k in range(A.shape[1])])
    norm = np.sqrt(np.dot(z[1:], z[1:]))
    return z / norm


def compute_qmatrix(xs: np.ndarray) -> np.ndarray:
    ps = np.array([compute_plane(x) for x in xs])
    # TODO: Check the axis!!
    return np.sum(np.array([np.outer(p, p) for p in ps]), axis=0)


class Simplification:
    def __init__(self, points: np.ndarray, triangles: np.ndarray):
        pass

    @property
    def topology(self):
        return self._topology

    @property
    def heap(self):
        pass

    def best_edge(self):
        pass

    def step(self):
        pass
