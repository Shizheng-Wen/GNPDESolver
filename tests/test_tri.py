import sys 
sys.path.append("..")
import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.tri import triplot
from src.graph import Mesh, delaunay, tri_medians, delaunay_edges

def test_tri_medians():

    def _compute_triangulation_medians(tri: np.ndarray, points: np.ndarray) -> np.ndarray:
        edges = np.zeros(shape=tri.shape)
        medians = np.zeros(shape=tri.shape)
        for i in range(tri.shape[1]):
            _points = points[np.delete(tri, i, axis=1)]
            _points = [p.squeeze(1) for p in np.split(_points, axis=1, indices_or_sections=2)]
            edges[:, i] = np.linalg.norm(np.subtract(*_points), axis=1)
        for i in range(tri.shape[1]):
            medians[:, i] = .67 * np.sqrt((2 * np.sum(np.power(np.delete(edges, i, axis=1), 2), axis=1) - np.power(edges[:, i], 2)) / 4)
        return medians


    mesh = Mesh.grid()
    triangles = delaunay(mesh.points)
    medians = tri_medians(mesh.points[triangles])
    
    points = mesh.points.numpy()
    triangles = triangles.numpy()
    medians = medians.numpy()
    medians_ = _compute_triangulation_medians(triangles, points)
    np.testing.assert_allclose(medians, medians_)   


if __name__ == '__main__':
    test_tri_medians()

