import pytest
import numpy as np
import numpy.ma as ma
from zmsh.simplicial import Topology, incidence, oriented


def test_incidences():
    A = np.array([0, 1, 2])
    assert incidence(A, [1, 2]) == +1
    assert incidence(A, [0, 2]) == -1
    assert incidence(A, [0, 1]) == +1
    assert incidence(A, [2, 0]) == +1

    B = np.array([3, 2, 1])
    assert oriented(A, B)
    assert not oriented(A, np.flip(B))
    assert oriented(A, np.roll(B, 1))
    assert oriented(A, np.roll(B, 2))

    C = np.array([0, 3])
    assert incidence(A, C) == 0


def test_non_orientable():
    topology = np.array([[0, 1, 2], [1, 2, 3]])
    assert not oriented(topology[0], topology[1])
