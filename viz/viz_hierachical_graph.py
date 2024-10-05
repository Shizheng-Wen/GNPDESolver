import os
import sys 
sys.path.append("..")
import numpy as np  
import torch
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection, PatchCollection
from src.graph import Mesh, \
                    minimal_support, \
                    subsample, \
                    radius_bipartite_graph, \
                    hierarchical_graph,\
                    domain_shifts,\
                    shift

def viz_hierachical_graph():
    mesh = Mesh.grid()
    points = mesh.points
    points = points[torch.randperm(points.shape[0])]
    level = 4

    fig, ax = plt.subplots(ncols=level, figsize=(8*level,8))
    norm = mcolors.Normalize(vmin=0, vmax=level-1)  # Normalize from 0 to level-1
    # color as c
    cmap = plt.get_cmap("viridis")
    cmap = plt.get_cmap("jet")
    for l in range(level):
        edges, domains, levels = hierarchical_graph(points, level=l+1, return_levels=True)
        edges = edges.T

        ax[l].scatter(points[:,0], points[:,1], color="blue", label="points")
        ax[l].add_collection(LineCollection(points[edges], color="dodgerblue", alpha=0.5, label=f"level {l}"))
        ax[l].legend()
        ax[l].set_title(f"level {l}")
    
    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/hierachical_graph.png")
    # plt.show()


def viz_hierachical_graph_periodic():
    mesh = Mesh.grid(periodic=True)
    points = mesh.points
    points = points[torch.randperm(points.shape[0])]
    level = 2

    _domain_shifts = domain_shifts((2.0, 2.0))

    edges, domains, levels = hierarchical_graph(points, level=level, return_levels=True, domain_shifts=_domain_shifts)
    
    fig, axes = plt.subplots(ncols=level, nrows=2, figsize=(8*level,8*2))

    shifted_points = shift(points, _domain_shifts)
    for l in range(level):

        axes[0, l].set_title(f"level {l}")
         
        axes[0, l].scatter(shifted_points[:,0], shifted_points[:,1], color="gray", label="shifed", alpha=0.5, s=5)
        axes[0, l].scatter(points[:,0], points[:,1], color="blue", label="points", s=10)        
        axes[0, l].axhline(-1, color="black", linestyle="--")
        axes[0, l].axhline(1, color="black", linestyle="--")
        axes[0, l].axvline(-1, color="black", linestyle="--")
        axes[0, l].axvline(1, color="black", linestyle="--")

        _edges = edges[:,levels == l]
        _domains = domains[:,levels == l]
        shifted_edges = points[_edges] + _domain_shifts[_domains]

        is_center = (_domains == 0).all(0)
    
        axes[0, l].add_collection(LineCollection(shifted_edges[:,is_center,:].transpose(1,0), color="dodgerblue", alpha=0.5, label=f"center edges"))
        axes[0, l].add_collection(LineCollection(shifted_edges[:,~is_center,:].transpose(1,0), color="violet", alpha=0.5, label=f"ghost edges"))

        axes[1, l].scatter(points[:,0], points[:,1], color="blue", label="points", s=10)
        
        axes[1, l].add_collection(LineCollection(points[_edges][:, is_center, :].transpose(1,0), color="dodgerblue", alpha=0.5, label=f"center edges"))
        axes[1, l].add_collection(LineCollection(points[_edges][:, ~is_center, :].transpose(1,0), color="violet", alpha=0.5, label=f"ghost edges"))


    os.makedirs("../outputs/viz", exist_ok=True)
    fig.savefig(f"../outputs/viz/hierachical_graph_periodic.png")


if __name__ == "__main__":
    viz_hierachical_graph_periodic()