import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits import mplot3d
import zmsh


def test_plotting_triangle():
    topology = zmsh.examples.simplex(2)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    geometry = zmsh.Geometry(topology, points)

    fig, ax = plt.subplots()
    zmsh.visualize(geometry, dimension=2, colors="tab:grey", ax=ax)
    zmsh.visualize(geometry, dimension=1, ax=ax)
    pointcolors = ["tab:blue", "tab:green", "tab:orange"]
    zmsh.visualize(geometry, dimension=0, colors=pointcolors, ax=ax)


def test_plotting_empty_cell():
    topology = zmsh.examples.simplex(2)

    polygons = topology.cells(2)
    polygons.resize(2)

    edges = topology.cells(1)
    edges.resize(4)

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    geometry = zmsh.Geometry(topology, points)

    edgecolors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    facecolors = ["tab:grey", "tab:cyan"]

    fig, ax = plt.subplots()
    zmsh.visualize(geometry, dimension=2, colors=facecolors, ax=ax)
    zmsh.visualize(geometry, dimension=1, colors=edgecolors, ax=ax)
    zmsh.visualize(geometry, dimension=0, ax=ax)


def test_plotting_tetrahedron():
    topology = zmsh.examples.simplex(3)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    geometry = zmsh.Geometry(topology, points)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((-0.5, 1.5))
    ax.set_zlim((-0.5, 1.5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    alpha = 0.5
    collection = zmsh.visualize(
        geometry, dimension=2, colors=colors, alpha=alpha, ax=ax
    )
    assert collection.get_alpha() == alpha
    with tempfile.NamedTemporaryFile(suffix=".png") as output_file:
        fig.savefig(output_file.name, dpi=100)
    assert len(collection.get_paths()) == len(topology.cells(2))


def test_polygon_plotting():
    topology = zmsh.Topology(dimension=2, num_cells=[6, 7, 2])
    edges = topology.cells(1)
    edges[0] = (0, 1), (-1, +1)
    edges[1] = (1, 2), (-1, +1)
    edges[2] = (0, 3), (-1, +1)
    edges[3] = (1, 4), (-1, +1)
    edges[4] = (2, 5), (-1, +1)
    edges[5] = (3, 4), (-1, +1)
    edges[6] = (4, 5), (-1, +1)

    polygons = topology.cells(2)
    polygons[0] = (0, 2, 3, 5), (+1, -1, +1, -1)
    polygons[1] = (1, 3, 4, 6), (+1, -1, +1, -1)

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    geometry = zmsh.Geometry(topology, points)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    colors = ["tab:blue", "tab:green"]
    zmsh.visualize(geometry, dimension=2, colors=colors, ax=ax)
    zmsh.visualize(geometry, dimension=1, ax=ax)
