import abc
import collections
import numbers
import operator
import numpy as np
from scipy.sparse import dok_matrix


# Hack to make getting faces of 0-dimensional cells work nice; the boundary
# matrix needs to have a `toarray` method.
class SparseView(np.ndarray):
    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def toarray(self):
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

    def __getitem__(self, key):
        r"""Get the faces and corresponding signs of a particular cell"""
        face_ids = np.array(sorted(list(set(self._matrix[:, key].nonzero()[0]))))
        signs = np.array(self._matrix[face_ids, :][:, key].todense())
        if isinstance(key, numbers.Integral):
            signs = signs.flatten()
        return face_ids, signs

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

    def closure(self, key):
        cell_ids, matrices = [key], []
        for d in range(self._dimension, self._topology.dimension):
            cocells = self._topology.cocells(d)
            coface_ids, signs = cocells[cell_ids[-1]]
            cell_ids.append(coface_ids)
            matrices.append(signs)

        return cell_ids, matrices


class Cells(CellView):
    def __init__(self, topology, dimension):
        r"""A view of the cells of a particular dimension of a topology"""
        super().__init__(topology, dimension)

    @property
    def _matrix(self):
        r"""The matrix representing the boundary operator on chains"""
        return self._topology._boundaries[self._dimension]

    def __setitem__(self, key, value):
        r"""Set the faces and corresponding signs of a particular cell"""
        rows, cols = key
        self._matrix[:, cols] = 0

        if isinstance(cols, numbers.Integral):
            self._matrix[rows, cols] = value
        elif isinstance(cols, (collections.abc.Sequence, np.ndarray)):
            for index, col in enumerate(cols):
                self._matrix[rows, col] = value[:, index]
        elif isinstance(cols, slice):
            r = range(cols.start or 0, cols.stop or len(self), cols.step or 1)
            for index, col in enumerate(r):
                self._matrix[rows, col] = value[:, index]
        else:
            raise TypeError(
                "key %s has type %s, must be a number, sequence, array, or slice!"
                % (key, type(key))
            )

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

    def permute(self, permutation):
        r"""Re-number the cells of the topology"""
        topology, dimension = self._topology, self._dimension
        topology._boundaries[dimension] = self._matrix[:, permutation]
        if dimension < topology.dimension:
            matrix = self._topology._boundaries[dimension + 1]
            self._topology._boundaries[dimension + 1] = matrix[permutation, :]

    def remove_empty_cells(self):
        r"""Re-number the cells of the topology and resize it so that there are
        no empty cells"""
        data = np.array(
            sorted(
                [
                    (index, len(face_ids) == 0)
                    for index, (face_ids, signs) in enumerate(self)
                ],
                key=operator.itemgetter(1),
            )
        )
        num_nonempty_cells = (~data[:, 1].astype(bool)).sum()
        if num_nonempty_cells < len(self):
            permutation = data[:, 0]
            self.permute(permutation)
            self.resize(num_nonempty_cells)

    def closure(self, key):
        cell_ids, matrices = [key], []
        for d in range(self._dimension, -1, -1):
            cells = self._topology.cells(d)
            face_ids, signs = cells[cell_ids[-1]]
            cell_ids.append(face_ids)
            matrices.append(signs)

        if len(matrices[0].shape) == 1:
            matrices[0] = matrices[0].reshape((-1, 1))

        return cell_ids[::-1][1:], matrices[::-1]


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
