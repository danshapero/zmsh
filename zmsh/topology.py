import abc
import numpy as np
from scipy.sparse import dok_matrix


# Hack to make getting faces of 0-dimensional cells work nice; the boundary
# matrix needs to have a `todense` method.
class SparseView(np.ndarray):
    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def todense(self):
        return self


class CellView(abc.ABC):
    def __init__(self, topology, dimension):
        self._topology = topology
        self._dimension = dimension

    @property
    @abc.abstractmethod
    def _matrix(self):
        pass

    def __len__(self):
        return self._matrix.shape[1]

    def __getitem__(self, index):
        r"""Get the faces and corresponding signs of a particular cell"""
        faces = self._matrix[:, index].nonzero()[0]
        signs = np.array(self._matrix[faces, index].todense()).flatten()
        return faces, signs

    def __iter__(self):
        r"""Iterate over all the cells of a given dimension"""
        return (self[index] for index in range(len(self)))


class CoCells(CellView):
    def __init__(self, topology, dimension):
        r"""A view of the cocells of a particular dimension of a topology"""
        super().__init__(topology, dimension)

    @property
    def _matrix(self):
        r"""The matrix representing the coboundary operator on chains"""
        return self._topology._boundaries[self._dimension + 1].T


class Cells(CellView):
    def __init__(self, topology, dimension):
        r"""A view of the cells of a particular dimension of a topology"""
        super().__init__(topology, dimension)

    @property
    def _matrix(self):
        r"""The matrix representing the boundary operator on chains"""
        return self._topology._boundaries[self._dimension]

    def __setitem__(self, index, value):
        r"""Set the faces and corresponding signs of a particular cell"""
        faces, signs = value
        self._matrix[:, index] = 0
        self._matrix[faces, index] = signs

    def resize(self, size):
        if self._dimension == 0:
            dtype = self._matrix.dtype
            self._topology._boundaries[0] = np.ones((1, size), dtype=dtype)
        else:
            shape = self._matrix.shape
            self._matrix.resize((shape[0], size))

        if self._dimension < self._topology.dimension:
            shape = self._topology._boundaries[self._dimension + 1].shape
            self._topology._boundaries[self._dimension + 1].resize((size, shape[1]))


class Topology:
    def __init__(self, dimension, num_cells=None, **kwargs):
        if num_cells is None:
            num_cells = [0] * (dimension + 1)

        dtype = kwargs.get("dtype", np.int8)
        mat_type = kwargs.get("mat_type", dok_matrix)
        self._boundaries = [SparseView(np.ones((1, num_cells[0]), dtype=dtype))] + [
            mat_type((num_cells[d - 1], num_cells[d]), dtype=dtype)
            for d in range(1, dimension + 1)
        ]

    @property
    def dimension(self):
        r"""The max dimension of all cells of the topology"""
        return len(self._boundaries) - 1

    def cells(self, dimension):
        r"""Get the cells of the given dimension"""
        return Cells(self, dimension)

    def cocells(self, dimension):
        r"""Get the cocells of the given dimension"""
        return CoCells(self, dimension)

    def boundary(self, dimension):
        r"""Get the boundary matrix from the space of `k`-dimensional chains
        down to the space of `k - 1`-dimensional chains"""
        return self._boundaries[dimension]

    def coboundary(self, dimension):
        r"""Get the coboundary matrix from the space of `k`-dimensional chains
        up to the space of `k + 1`-dimensional chains"""
        return self._boundaries[dimension + 1].T
