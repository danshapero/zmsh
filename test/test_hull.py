import itertools
import numpy as np
from numpy import linalg, random
import zmsh

def permute_eq(A, B):
    if A.shape != B.shape:
        return False

    N = A.shape[1]
    for p in itertools.permutations(list(range(N)), N):
        diff = linalg.norm(A - B[:, p], ord=1)
        if diff == 0:
            return True

    return False


def test_hull():
    points = np.array(
        [[0., 0.],
         [1., 0.],
         [1., 1.],
         [0., 1.],
         [.5, .5]]
    )

    hull_machine = zmsh.ConvexHull(points)
    topology = hull_machine.run()
    delta = topology.boundary(dimension=1).todense()

    delta_true = np.array(
        [[-1, +1, 0, 0, 0],
         [0, -1, +1, 0, 0],
         [0, 0, -1, +1, 0],
         [+1, 0, 0, -1, 0]],
        dtype=np.int8
    ).T

    assert permute_eq(delta, delta_true)
