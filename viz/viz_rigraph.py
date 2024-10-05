import os
import sys 
sys.path.append("..")
import matplotlib.pyplot as plt 

from src.graph import Mesh, RegionInteractionGraph


def viz_rigraph():
    mesh = Mesh.grid()
    rigraph = RegionInteractionGraph.from_mesh(mesh)
    ppos, rpos = rigraph.physical_to_regional.get_ndata("pos")


    ppos, rpos = ppos.numpy(), rpos.numpy()

    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(ppos[:,0], ppos[:,1], color="dodgerblue", label="physical", alpha=0.5)
    ax.scatter(rpos[:,0], rpos[:,1], color="orange", label="regional", alpha=0.5)

    ax.legend()
    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/rigraph.png")



if __name__ == '__main__':
    viz_rigraph()