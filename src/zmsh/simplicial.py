from __future__ import annotations
import numpy as np
import numpy.ma as ma
import numbers


Topology = np.ndarray


def get_face_index_in_cell(cell: np.ndarray, face: np.ndarray) -> int:
    if len(cell) != len(face) + 1:
        raise ValueError(
            "Input arrays must have size `d + 1` and `d`, got %d and %d"
            % (len(cell), len(face))
        )
    for index in range(len(cell)):
        if np.isin(face, np.delete(cell, index)).all():
            return index


def parity(permutation: np.ndarray) -> int:
    # TODO: This is not very efficient
    num_transpositions = sum(
        1
        for index1, value1 in enumerate(permutation)
        for index2, value2 in enumerate(permutation)
        if index1 < index2 and value1 > value2
    )
    return +1 if (num_transpositions % 2 == 0) else -1


def incidence(cell: np.ndarray, face: np.ndarray) -> int:
    r"""Return the sign (-1, 0, +1) of how the k-simplex `face` is attached in
    the (k + 1)-simplex `cell`"""
    index = get_face_index_in_cell(cell, face)
    if index is None:
        return 0
    cface = list(np.delete(cell, index))
    permutation = np.array([cface.index(vertex_id) for vertex_id in face])
    return (-1) ** index * parity(permutation)


def oriented(cell1: np.ndarray, cell2: np.ndarray) -> bool:
    r"""Return `true` if the two neighboring cells are properly oriented, i.e.
    they have opposite incidences w.r.t. their common face"""
    if len(cell1) != len(cell2):
        raise ValueError(
            "Can only check orientation of two cells of the same dimension!"
        )
    common_face = np.intersect1d(cell1, cell2)
    incidence1 = incidence(cell1, common_face)
    incidence2 = incidence(cell2, common_face)
    not_adjacent = incidence1 == 0
    opposites = incidence(cell1, common_face) != incidence(cell2, common_face)
    return not_adjacent or opposites
