import sys 
sys.path.append("..")
import numpy as np  
import torch
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.tri import triplot
from src.graph import Mesh, \
                    remove_duplicate_edges, \
                    sort_edges, \
                    domain_shifts,\
                    minimal_support, \
                    subsample, \
                    radius_bipartite_graph

def test_remove_duplicate_edges():
    edges = np.array([
        [0, 1],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [3, 0],
        [0, 2],
    ]).T
    target_edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2],
    ]).T
    edges = remove_duplicate_edges(torch.from_numpy(edges))
    edges = sort_edges(edges)
    target_edges = sort_edges(torch.from_numpy(target_edges))
    torch.testing.assert_close(edges, target_edges)
    

def test_domain_shifts():

    target_shifts = torch.from_numpy(np.mgrid[-2:4:2, -2:4:2].reshape(2, -1).T).long()

    shifts = domain_shifts(span=(2,2)).long()
    
    def _sort(x):
        arg = (x[:,0] + x[:,1]).argsort()
        return x[arg]
    shifts = _sort(shifts)
    target_shifts = _sort(target_shifts)
    torch.testing.assert_close(shifts, target_shifts)








    
if __name__ == "__main__":
    test_remove_duplicate_edges()
    test_domain_shifts()

