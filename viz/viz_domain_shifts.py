import os
import sys 
sys.path.append("..")
import numpy as np  
import matplotlib.pyplot as plt 
from src.graph import Mesh, \
                    domain_shifts,\
                    shift

def viz_domain_shifts():
    mesh = Mesh.grid()
    _domain_shifts = domain_shifts((2,2))

    fig, ax = plt.subplots(figsize=(8,8))
    points = mesh.points.numpy()
    spoints = shift(mesh.points, _domain_shifts).numpy()

    ax.plot(points[:,0], points[:,1], "o", label="physical points")
    ax.plot(spoints[:,0], spoints[:,1], "x", label="shifted points")
    ax.legend()
    ax.set_title("Domain Shifts")
    
    os.makedirs("../outputs/viz", exist_ok=True)
    plt.savefig(f"../outputs/viz/domain_shifts.png")

    # plt.show()

if __name__ == "__main__":
    viz_domain_shifts()
