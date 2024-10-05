import os
import sys 
sys.path.append("..")
import numpy as np  
import torch
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection, PatchCollection
from src.graph import Mesh, \
                    minimal_support, \
                    subsample, \
                    radius_bipartite_graph

def viz_radius_bipartite_graph():
    mesh = Mesh.grid()
    ppoints = mesh.points
    rpoints = torch.cat([
                ppoints[mesh.is_boundary],
                subsample(ppoints[~mesh.is_boundary], factor=0.3)
            ])
    radii = minimal_support(rpoints)
    edges = radius_bipartite_graph(ppoints, rpoints,  radii)
    src = ppoints[edges[0]]
    dst = rpoints[edges[1]]

    edges = torch.stack([src, dst], 1).numpy()
    ppoints = ppoints.numpy()
    rpoints = rpoints.numpy()
    radii = radii.numpy()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(ppoints[:,0], ppoints[:,1], color="blue", label="physical points")
    ax.scatter(rpoints[:,0], rpoints[:,1], color="orange", marker="x", label="regional points")
    patches = []
    for i, (x, y) in enumerate(rpoints):
        if i == 0:
            kwargs = {"label":"support"}
        else:
            kwargs = {}
        patches.append(Circle((x, y), radii[i],fill=False,**kwargs))
    pc = PatchCollection(patches, alpha=0.2,  facecolor='orange', linestyle='--', edgecolor='orange')
    ax.add_collection(pc)
    ax.add_collection(LineCollection(edges, color="green", label="edges", alpha=0.5))
    ax.legend()
    ax.set_title("Radius Bipartite Graph")
    
    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/radius_bipartite_graph.png")
    plt.show()


if __name__ == "__main__":
    viz_radius_bipartite_graph()

