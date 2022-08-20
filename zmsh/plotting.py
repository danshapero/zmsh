import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def _visualize_points(ax, geometry, **kwargs):
    kwargs["c"] = kwargs.pop("colors", "black")
    return ax.scatter(*geometry.points.T, **kwargs)


def _visualize_edges(ax, geometry, **kwargs):
    edges = geometry.topology.cells(1)
    segments = []

    colors = kwargs.pop("colors", "black")
    if matplotlib.colors.is_color_like(colors):
        colors = len(edges) * [colors]

    edgecolors = []
    for edge_index in range(len(edges)):
        vertex_indices = edges[edge_index][0]
        if len(vertex_indices) > 0:
            segments.append(geometry.points[vertex_indices])
            edgecolors.append(colors[edge_index])

    kwargs["edgecolors"] = edgecolors
    if geometry.dimension == 2:
        collection = LineCollection(segments, **kwargs)
    elif geometry.dimension == 3:
        collection = Line3DCollection(segments, **kwargs)
    else:
        raise ValueError("Geometric dimension must be 2 or 3!")

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def _get_vertices(topology, cell_index):
    polygons = topology.cells(2)
    edges = topology.cells(1)
    vertices = set()
    for edge_index in polygons[cell_index][0]:
        for vertex in edges[edge_index][0]:
            vertices.add(vertex)

    if len(vertices) > 0:
        return np.array(list(vertices))

    return []


def _visualize_polygons(ax, geometry, **kwargs):
    polygons = geometry.topology.cells(2)
    edges = geometry.topology.cells(1)
    verts = []

    colors = kwargs.pop("colors", matplotlib.colors.to_rgb("tab:blue"))
    if matplotlib.colors.is_color_like(colors):
        colors = len(polygons) * [colors]

    poly_facecolors = []
    for cell_index in range(len(polygons)):
        vertex_indices = _get_vertices(geometry.topology, cell_index)
        count = 0
        if len(vertex_indices) == 3:
            points = geometry.points[vertex_indices]
            verts.append(points)
            count = 1
        elif len(vertex_indices) > 3:
            centroid = geometry.points[vertex_indices].mean(axis=0)
            edge_indices = polygons[cell_index][0]
            for edge_index in edge_indices:
                vertex_indices = edges[edge_index][0]
                points = geometry.points[vertex_indices]
                verts.append(np.vstack((points, centroid)))

            count = len(edge_indices)

        poly_facecolors.extend(count * [colors[cell_index]])

    kwargs["edgecolors"] = kwargs.get("edgecolors", None)
    kwargs["facecolors"] = poly_facecolors

    if geometry.dimension == 2:
        collection = PolyCollection(verts, **kwargs)
    elif geometry.dimension == 3:
        collection = Poly3DCollection(verts, **kwargs)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def visualize(geometry, dimension, **kwargs):
    r"""Visualize one of the skeletons of the geometry

    Parameters
    ----------
    geometry : zmsh.Geometry
        The manifold to visualize
    dimension : int
        Which cells to visualize -- 0 for vertices, 1 for edges, 2 for polygons

    Other Parameters
    ----------------
    colors : color or list of colors
        Passed to `LineCollection` or `PolyCollection` to determine the color
        of each cell of the given dimension. If you supply a list of colors,
        the order of colors in the list corresponds directly to the numbering
        of the cells
    **kwargs : dict
        Passed to `LineCollection` or `PolyCollection`

    Notes
    -----
    Visualizing both edges and polygons works pretty badly for 3D geometries
    due to fundamental limitations in the `mpl_toolkits.mplot3d` module
    related to, for example, a line being partially occluded by a polygon.
    """
    if (geometry.dimension > 3) or dimension >= 3:
        raise NotImplementedError(
            "Don't know how to visualize higher-dimensional things yet!"
        )

    try:
        ax = kwargs.pop("ax")
    except KeyError:
        kw = {"projection": "3d"} if geometry.dimension == 3 else {}
        fig = plt.figure()
        ax = fig.add_subplot(**kw)

    functions = [_visualize_points, _visualize_edges, _visualize_polygons]
    return functions[dimension](ax, geometry, **kwargs)
