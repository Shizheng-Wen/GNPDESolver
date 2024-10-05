import os
import sys 
sys.path.append("..")
import numpy as np  
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from src.graph import Mesh, \
                    minimal_support \

def viz_minimal_support():
    mesh = Mesh.grid()
    radii = minimal_support(mesh.points)

    fig, ax = plt.subplots(figsize=(8,8))

    points = mesh.points.numpy()
    radii = radii.numpy()
    
    patches = []
    for i, (x, y) in enumerate(points):
        if i == 0:
            kwargs = {"label":"support"}
        else:
            kwargs = {}
        patches.append(Circle((x, y), radii[i],fill=False,**kwargs))
    pc = PatchCollection(patches, alpha=0.4,  facecolor='orange', linestyle='--', edgecolor='orange')
    ax.add_collection(pc)

    ax.scatter(points[:,0], points[:,1], color="blue", label="points")
    ax.legend()
    ax.set_title("Minimal Support")

    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/minimal_support.png")

    # plt.show()


if __name__ == "__main__":
    viz_minimal_support()

