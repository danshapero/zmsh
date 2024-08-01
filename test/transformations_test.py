import pytest
import numpy as np
import zmsh
from zmsh.examples import simplex, cube


@pytest.mark.parametrize(
    "fn, dimension", [(simplex, 2), (simplex, 3), (cube, 2), (cube, 3)]
)
def test_splitting_single_polygon(fn, dimension):
    geometry = fn(dimension)
    topology = geometry.topology
    cells_ids, Ds = topology.cells(topology.dimension).closure(0)
    Es = zmsh.transformations.split(Ds)
    for E_1, E_2 in zip(Es[:-1], Es[1:]):
        assert np.linalg.norm(E_1 @ E_2) == 0

    C = Es[-1] @ np.ones(Es[-1].shape[1], dtype=np.int8)
    assert np.array_equal(C[: Ds[-1].shape[0]], Ds[-1].flatten())


def test_splitting_multiple_polygons():
    topology = zmsh.Topology(dimension=2, num_cells=[4, 5, 2])

    edges = topology.cells(1)
    edges[:, :] = np.array(
        [
            [-1, 0, +1, -1, 0],
            [+1, -1, 0, 0, +1],
            [0, +1, -1, 0, 0],
            [0, 0, 0, +1, -1],
        ],
        dtype=np.int8,
    )

    triangles = topology.cells(2)
    triangles[:, :] = np.array(
        [[+1, -1], [+1, 0], [+1, 0], [0, +1], [0, +1]], dtype=np.int8
    )

    cells_ids, Ds = triangles.closure([0, 1])
    Es = zmsh.transformations.split(Ds)
    for E_1, E_2 in zip(Es[:-1], Es[1:]):
        assert np.linalg.norm(E_1 @ E_2) == 0

    C = Es[-1] @ np.ones(Es[-1].shape[1], dtype=np.int8)
    C_expected = np.zeros_like(C)
    C_expected[: Ds[-1].shape[0]] = Ds[-1].sum(axis=1)
    assert np.array_equal(C, C_expected)


def split_poly_fuzzer(rng):
    min_size, max_size = 3, 16
    n = rng.integers(min_size, max_size, endpoint=True)
    num_cells = [n, n, 1]
    topology = zmsh.Topology(dimension=2, num_cells=num_cells)

    # Create the canonical circulant matrix
    # [
    #     [-1, 0,  ..., +1],
    #     [+1, -1, ...,  0],
    #     ...
    #     [ 0, 0,  ..., -1],
    # ]
    # describing the connectivity of an `n`-gon
    D = (np.diag(np.ones(n - 1), -1) - np.eye(n)).astype(np.int8)
    D[0, -1] = +1

    # Randomly permute the numbering of the polygon edges
    D = D[:, rng.permutation(n)]

    # Randomly switch the signs of some of the edges
    S = (2 * rng.integers(0, 1, size=n, endpoint=True) - 1).astype(np.int8)
    D = D @ np.diag(S)

    edges = topology.cells(1)
    edges[:, :] = D
    polys = topology.cells(2)
    polys[:, 0] = S

    D_1, D_2 = topology.boundary(1), topology.boundary(2)
    assert np.linalg.norm((D_1 @ D_2).toarray()) == 0

    cell_ids, Ds = topology.cells(2).closure(0)
    Es = zmsh.transformations.split(Ds)

    for E_1, E_2 in zip(Es[:-1], Es[1:]):
        assert np.linalg.norm(E_1 @ E_2) == 0

    C = Es[-1] @ np.ones(Es[-1].shape[1], dtype=np.int8)
    assert np.array_equal(C[: Ds[-1].shape[0]], Ds[-1].flatten())


def test_splitting_random_polygons():
    rng = np.random.default_rng(seed=1729)
    num_trials = 20
    for trial in range(num_trials):
        split_poly_fuzzer(rng)


def test_identifying_separators():
    topology = zmsh.Topology(dimension=2, num_cells=[5, 6, 1])
    edges = topology.cells(1)
    edges[:, :] = np.array(
        [
            [-1, 0, 0, +1, 0, 0],
            [+1, -1, 0, 0, 0, 0],
            [0, +1, -1, 0, 0, 0],
            [0, 0, +1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )

    polygons = topology.cells(2)
    polygons[:, :] = np.array([[+1], [+1], [+1], [+1], [0], [0]], dtype=np.int8)

    Ds = [topology.boundary(k).toarray() for k in range(3)]
    assert not zmsh.transformations.identify_separators(Ds)
    components = zmsh.transformations.identify_components(Ds[-2], list(range(6)))
    assert len(components) == 1
    assert np.array_equal(components[0], np.arange(4))

    Ds[-2][:, 5] = np.array([-1, 0, 0, 0, +1], dtype=np.int8)
    assert len(zmsh.transformations.identify_components(Ds[-2], [5])) == 1
    assert not zmsh.transformations.identify_separators(Ds)

    Ds[-2][:, 4] = np.array([0, +1, 0, -1, 0], dtype=np.int8)
    null_components = zmsh.transformations.identify_components(Ds[-2], [4, 5])
    assert len(null_components) == 2
    complements = zmsh.transformations.identify_components(Ds[-2], [0, 1, 2, 3], [1, 3])
    assert len(complements) == 2
    separators = zmsh.transformations.identify_separators(Ds)
    assert len(separators) == 1
    separator, remainders = separators[0]
    assert np.array_equal(separator, np.array([4]))
    assert len(remainders) == 2


def test_identifying_separators_degenerate():
    topology = zmsh.Topology(dimension=2, num_cells=[2, 3, 1])
    edges = topology.cells(1)
    edges[:, :] = np.array([[-1, +1, -1], [+1, -1, +1]], dtype=np.int8)
    polygons = topology.cells(2)
    polygons[:, :] = np.array([[+1], [+1], [0]], dtype=np.int8)

    Ds = [topology.boundary(k).toarray() for k in range(3)]
    separators = zmsh.transformations.identify_separators(Ds)
    assert len(separators) == 1
    separator, remainders = separators[0]
    assert np.array_equal(separator, np.array([2]))


def test_bisecting_quadrilateral():
    topology = zmsh.Topology(dimension=2, num_cells=[4, 5, 1])
    edges = topology.cells(1)
    edges[:, :] = np.array(
        [
            [-1, 0, 0, +1, 0],
            [+1, -1, 0, 0, +1],
            [0, +1, -1, 0, 0],
            [0, 0, +1, -1, -1],
        ],
        dtype=np.int8,
    )

    polygons = topology.cells(2)
    polygons[:, :] = np.array(
        [[+1], [+1], [+1], [+1], [0]],
        dtype=np.int8,
    )

    Ds = [topology.boundary(k).toarray() for k in range(3)]
    E = zmsh.transformations.bisect(Ds)

    E_expected = np.array(
        [
            [+1, 0],
            [0, +1],
            [0, +1],
            [+1, 0],
            [-1, +1],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(E, E_expected)


def test_larger_bisector():
    topology = zmsh.Topology(dimension=2, num_cells=[5, 6, 1])
    edges = topology.cells(1)
    edges[:, :] = np.array(
        [
            [-1, 0, 0, +1, 0, 0],
            [+1, -1, 0, 0, 0, +1],
            [0, +1, -1, 0, 0, 0],
            [0, 0, +1, -1, -1, 0],
            [0, 0, 0, 0, +1, -1],
        ],
        dtype=np.int8,
    )

    polygons = topology.cells(2)
    polygons[:, :] = np.array(
        [[+1], [+1], [+1], [+1], [0], [0]],
        dtype=np.int8,
    )

    Ds = [topology.boundary(k).toarray() for k in range(3)]
    E = zmsh.transformations.bisect(Ds)

    E_expected = np.array(
        [
            [+1, 0],
            [0, +1],
            [0, +1],
            [+1, 0],
            [-1, +1],
            [-1, +1],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(E, E_expected)


def test_bisecting_hanging_edge():
    topology = zmsh.Topology(dimension=2, num_cells=[5, 6, 1])
    edges = topology.cells(1)
    edges[:, :] = np.array(
        [
            [-1, 0, 0, +1, 0, -1],
            [+1, -1, 0, 0, +1, 0],
            [0, +1, -1, 0, 0, 0],
            [0, 0, +1, -1, -1, 0],
            [0, 0, 0, 0, 0, +1],
        ],
        dtype=np.int8,
    )

    polygons = topology.cells(2)
    polygons[:, 0] = np.array([+1, +1, +1, +1, 0, 0], dtype=np.int8)

    Ds = [topology.boundary(k).toarray() for k in range(3)]
    separators = zmsh.transformations.identify_separators(Ds)
    assert len(separators) == 1
    separator, remainders = separators[0]
    assert np.array_equal(separator, np.array([4]))
    E = zmsh.transformations.bisect(Ds)

    E_expected = np.array(
        [
            [+1, 0],
            [0, +1],
            [0, +1],
            [+1, 0],
            [-1, +1],
            [0, 0],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(E, E_expected)
