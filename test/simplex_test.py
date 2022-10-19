from itertools import permutations, product
import numpy as np
import zmsh


def test_simplex_to_chain_complex():
    max_dim = 5
    for dimension in range(1, max_dim + 1):
        # Check that `∂∂ == 0`
        As = zmsh.simplicial.simplex_to_chain_complex(dimension)
        for A_1, A_2 in zip(As[:-1], As[1:]):
            assert np.linalg.norm(A_1 @ A_2) == 0

        # Check that every vertex is connected to every other vertex
        A = As[1]
        assert np.all(A @ A.T != 0)


def test_matrix_transformations():
    A = np.array([[+1, 0, -1], [-1, +1, 0], [0, -1, +1]], dtype=np.int8)
    p_exact = np.array([2, 1, 0], dtype=int)
    s_exact = np.array([+1, -1, +1], dtype=int)
    B = A[:, p_exact] @ np.diag(s_exact)
    p, s = zmsh.simplicial.compute_matrix_transformation(A, B)
    assert np.array_equal(p, p_exact)
    assert np.array_equal(s, s_exact)

    A = np.array(
        [[-1, 0, 0, +1], [+1, -1, 0, 0], [0, +1, -1, 0], [0, 0, +1, -1]], dtype=np.int8
    )
    p_exact = np.array([3, 1, 2, 0], dtype=int)
    s_exact = np.array([+1, -1, -1, +1], dtype=int)
    B = A[:, p_exact] @ np.diag(s_exact)
    p, s = zmsh.simplicial.compute_matrix_transformation(A, B)
    assert np.array_equal(p, p_exact)
    assert np.array_equal(s, s_exact)


def test_simplicial_chain_complex_morphism():
    As = zmsh.simplicial.simplex_to_chain_complex(2)
    Bs = [np.copy(A) for A in As]
    As[1][:, 1] *= -1
    As[2][1, :] *= -1
    ps, ss = zmsh.simplicial.compute_morphism(As[1:], Bs[1:])
    assert np.array_equal(ss[0], [+1, -1, +1])


def test_triangle_orientation_exhaustive():
    A_0 = np.array([[+1], [+1], [+1]], dtype=np.int8)
    A_1 = np.array([[-1, 0, +1], [+1, -1, 0], [0, +1, -1]], dtype=np.int8)
    A_2 = np.array([[+1], [+1], [+1]], dtype=np.int8)

    for p, s in zip(permutations(range(3)), product([+1, -1], repeat=3)):
        B_0 = np.copy(A_0)
        B_1 = A_1[:, p] @ np.diag(s)
        B_2 = np.diag(s) @ A_2[p, :]
        assert zmsh.simplicial.orientation([B_0, B_1, B_2]) == +1

        B_2 = -np.diag(s) @ A_2[p, :]
        assert zmsh.simplicial.orientation([B_0, B_1, B_2]) == -1


def test_tetrahedron_orientation_random():
    rng = np.random.default_rng(seed=1729)
    A_0, A_1, A_2, A_3 = zmsh.simplicial.simplex_to_chain_complex(3)

    num_trials = 50
    for trial in range(num_trials):
        p_1 = rng.permutation(A_1.shape[1])
        p_2 = rng.permutation(A_2.shape[1])

        s_1 = rng.choice([+1, -1], size=A_1.shape[1])
        s_2 = rng.choice([+1, -1], size=A_2.shape[1])

        B_0 = np.copy(A_0)
        B_1 = A_1[:, p_1] @ np.diag(s_1)
        B_2 = np.diag(s_1) @ A_2[p_1, :][:, p_2] @ np.diag(s_2)
        B_3 = np.diag(s_2) @ A_3[p_2, :]

        assert zmsh.simplicial.orientation([B_0, B_1, B_2, B_3]) == +1
        assert zmsh.simplicial.orientation([B_0, B_1, B_2, -B_3]) == -1


def test_topology_orientation():
    geometry = zmsh.examples.simplex(2)
    topology = geometry.topology
    cell_ids, Ds = topology.cells(2).closure(0)
    Ds[-1] = Ds[-1].reshape((3, 1))
    assert zmsh.simplicial.orientation(Ds) == +1
