import numpy as np
import zmsh


def test_splitting_multiple_polygons():
    d_0 = np.ones((1, 4), dtype=np.int8)
    d_1 = np.array(
        [
            [-1, 0, +1, -1, 0],
            [+1, -1, 0, 0, +1],
            [0, +1, -1, 0, 0],
            [0, 0, 0, +1, -1],
        ],
        dtype=np.int8,
    )
    d_2 = np.array([[+1, -1], [+1, 0], [+1, 0], [0, +1], [0, +1]], dtype=np.int8)
    Ds = [d_0, d_1, d_2]
    Es = zmsh.polytopal.vertex_split(Ds)
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
    d_0 = np.ones((1, n), dtype=np.int8)

    # Create the canonical circulant matrix
    # [
    #     [-1, 0,  ..., +1],
    #     [+1, -1, ...,  0],
    #     ...
    #     [ 0, 0,  ..., -1],
    # ]
    # describing the connectivity of an `n`-gon
    d_1 = (np.diag(np.ones(n - 1), -1) - np.eye(n)).astype(np.int8)
    d_1[0, -1] = +1

    # Randomly permute the numbering of the polygon edges
    d_1 = d_1[:, rng.permutation(n)]

    # Randomly switch the signs of some of the edges
    S = (2 * rng.integers(0, 1, size=n, endpoint=True) - 1).astype(np.int8)
    d_1 = d_1 @ np.diag(S)

    d_2 = S.reshape((n, 1))

    assert np.linalg.norm(d_1 @ d_2) == 0

    Ds = [d_0, d_1, d_2]
    Es = zmsh.polytopal.vertex_split(Ds)

    for E_1, E_2 in zip(Es[:-1], Es[1:]):
        assert np.linalg.norm(E_1 @ E_2) == 0

    C = Es[-1] @ np.ones(Es[-1].shape[1], dtype=np.int8)
    assert np.array_equal(C[: Ds[-1].shape[0]], Ds[-1].flatten())


def test_splitting_random_polygons():
    rng = np.random.default_rng(seed=1729)
    num_trials = 20
    for trial in range(num_trials):
        split_poly_fuzzer(rng)
