import zmsh

def test_resizing():
    for dimension in range(1, 5):
        topology = zmsh.Topology(dimension)
        assert topology.dimension == dimension
        for d in range(dimension + 1):
            assert topology.num_cells(d) == 0

        for d in range(dimension + 1):
            topology.set_num_cells(d, dimension - d + 1)

        for d in range(dimension + 1):
            assert topology.num_cells(d) == dimension - d + 1


def test_edge():
    topology = zmsh.Topology(1)
    topology.set_num_cells(0, 2)
    topology.set_num_cells(1, 1)

    topology.set_cell(1, 0, (0, 1), (-1, +1))
    products = topology.compute_nonzero_boundary_products()
    assert products == []


def test_bad_edge():
    topology = zmsh.Topology(1)
    topology.set_num_cells(0, 2)
    topology.set_num_cells(1, 1)

    topology.set_cell(1, 0, (0, 1), (+1, +1))
    products = topology.compute_nonzero_boundary_products()
    dimension, matrix = products[0]
    assert dimension == 0


def test_triangle():
    topology = zmsh.Topology(2)

    topology.set_num_cells(0, 3)
    topology.set_num_cells(1, 3)
    topology.set_num_cells(2, 1)

    topology.set_cell(1, 0, (0, 1), (-1, +1))
    topology.set_cell(1, 1, (1, 2), (-1, +1))
    topology.set_cell(1, 2, (2, 0), (-1, +1))

    topology.set_cell(2, 0, (0, 1, 2), (+1, +1, +1))

    products = topology.compute_nonzero_boundary_products()
    assert products == []


def test_bad_triangle():
    topology = zmsh.Topology(2)

    topology.set_num_cells(0, 3)
    topology.set_num_cells(1, 3)
    topology.set_num_cells(2, 1)

    topology.set_cell(1, 0, (0, 1), (-1, +1))
    topology.set_cell(1, 1, (1, 2), (-1, +1))
    topology.set_cell(1, 2, (2, 0), (-1, +1))

    topology.set_cell(2, 0, (0, 1, 2), (+1, -1, +1))

    products = topology.compute_nonzero_boundary_products()
    dimension, matrix = products[0]
    assert dimension == 1
