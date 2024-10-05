import os
import sys 
sys.path.append("..")
import numpy as np  
import matplotlib.pyplot as plt 
from src.graph import Mesh, delaunay, domain_shifts, shift


def viz_delaunay():
    mesh = Mesh.grid()
    triangles = delaunay(mesh.points)
    points = mesh.points.numpy()
    triangles = triangles.numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:,  0], points[:, 1], color="orange", label="points")
    ax.triplot(points[:, 0], points[:, 1], triangles, alpha=0.5, color="blue", label="Delaunay")
    ax.legend()
    ax.set_title("Delaunay Triangulation")

    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/delaunay.png")
    # plt.show()

def viz_delaunay_shifted():
    mesh = Mesh.grid(periodic=True)
    _domain_shifts = domain_shifts((2.0, 2.0))
    shifted_points = shift(mesh.points, _domain_shifts)
    triangles = delaunay(shifted_points)
    shifted_points = shifted_points.numpy()
    triangles = triangles.numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(shifted_points[:, 0], shifted_points[:, 1], color="gray", label="shifted points", s=3)
    ax.scatter(mesh.points[:, 0], mesh.points[:, 1], color="darkblue", label="points", s=5)
    ax.triplot(shifted_points[:, 0], shifted_points[:, 1], triangles, alpha=0.5, color="blue", label="Delaunay")
    ax.axvline(1.0, color="black", linestyle="--")
    ax.axvline(-1.0, color="black", linestyle="--")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.axhline(-1.0, color="black", linestyle="--")
    ax.legend()
    ax.set_title("Delaunay Triangulation Shifted")

    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/delaunay_shifted.png")

if __name__ == "__main__":
    viz_delaunay()
    viz_delaunay_shifted()
