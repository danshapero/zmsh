import numpy as np
from . import simplicial, polytopal


def compute_plane(x: np.ndarray) -> np.ndarray:
    pass


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
