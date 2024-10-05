import os
import sys 
sys.path.append("..")
import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from src.graph import Mesh, delaunay_edges, domain_shifts, shift

def viz_delaunay_edges():
    mesh = Mesh.grid()
    edges = delaunay_edges(mesh.points)
    edges = mesh.points[edges.T].numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mesh.points[:, 0], mesh.points[:, 1], color="orange", label="points")
    ax.add_collection(LineCollection(edges, color="blue", alpha=0.5))
    ax.legend()

    ax.set_title("Delaunay Edges")

    os.makedirs("../outputs/viz",exist_ok=True)
    plt.savefig("../outputs/viz/delaunay_edges.png")
    # plt.show()

def viz_delaunay_edges_shifted():
    mesh = Mesh.grid(periodic=True)
    _domain_shifts = domain_shifts((2.0, 2.0))
    shifted_points = shift(mesh.points, _domain_shifts)
    edges = delaunay_edges(shifted_points)
    edges_pos = shifted_points[edges.T]
    fig, ax = plt.subplots(figsize=(8, 8))

    is_center = (edges < mesh.points.shape[0]).all(axis=0)
    is_part_center = (edges < mesh.points.shape[0]).any(axis=0)
    
    ax.scatter(shifted_points[:, 0], shifted_points[:, 1], color="gray", label="shifted points", s=3)
    ax.scatter(mesh.points[:, 0], mesh.points[:, 1], color="darkblue", label="points", s=5)

    ax.add_collection(LineCollection(edges_pos[(~is_center)&(~is_part_center)], color="gray", alpha=0.5))
    ax.add_collection(LineCollection(edges_pos[is_center], color="dodgerblue", alpha=0.5))
    ax.add_collection(LineCollection(edges_pos[is_part_center], color="violet", alpha=0.5))
    ax.axvline(1.0, color="black", linestyle="--")
    ax.axvline(-1.0, color="black", linestyle="--")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.axhline(-1.0, color="black", linestyle="--")
    ax.legend()

    ax.set_title("Delaunay Edges Shifted")

    os.makedirs("../outputs/viz",exist_ok=True)
    plt.savefig("../outputs/viz/delaunay_edges_shifted.png")
    # plt.show()

if __name__ == '__main__':
    # viz_delaunay_edges()
    viz_delaunay_edges_shifted()