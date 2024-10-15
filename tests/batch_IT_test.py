import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

import sys
sys.path.append("../")
from src.model.cmpt.mlp import LinearChannelMLP
from src.model.cmpt.gno import IntegralTransform


from src.model.cmpt.utils.gno_utils import NeighborSearch

##########
# batchNeighborSearch
##########
class NeighborSearch_batch(nn.Module):
    """
    Neighborhood search between two arbitrary coordinate meshes with support for batched data.
    For each point `x` in `queries`, returns the indices of all points `y` in `data` 
    within the ball of radius `r` `B_r(x)`, using tensor operations without explicit loops.
    
    Parameters
    ----------
    use_open3d : bool
        Whether to use Open3D or native PyTorch implementation.
        NOTE: Open3D implementation requires 3D data.
    """
    def __init__(self, use_open3d=True):
        super().__init__()
        if use_open3d:
            from open3d.ml.torch.layers import FixedRadiusSearch
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = True
        else:
            self.search_fn = native_neighbor_search
            self.use_open3d = False

    def forward(self, data, queries, radius):
        """Find the neighbors in `data` for each point in `queries`
        within a ball of given radius. Returns outputs with batch dimension.

        Parameters
        ----------
        data : torch.Tensor of shape [n_batch, n, d]
            Search space of possible neighbors.
            NOTE: Open3D requires d=3.
        queries : torch.Tensor of shape [n_batch, m, d]
            Points for which to find neighbors.
            NOTE: Open3D requires d=3.
        radius : float
            Radius of each ball: B(queries[j], radius)
        
        Output
        ----------
        return_dict : dict
            Dictionary with keys: 'neighbors_index', 'neighbors_row_splits'
                neighbors_index: torch.Tensor of shape [n_batch, m, max_num_neighbors] with dtype=torch.int64
                    Neighbor indices for each batch and query, padded with -1.
                neighbors_row_splits: torch.Tensor of shape [n_batch, m+1] with dtype=torch.int64
                    Row splits indicating the cumulative count of neighbors per query in each batch.
        """
        if self.use_open3d:
            return_dict = open3d_neighbor_search(data, queries, radius)
        else:
            return_dict = self.search_fn(data, queries, radius)
        return return_dict

@torch.no_grad()
def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """
    Native PyTorch implementation of neighborhood search using tensor operations.
    Supports batched data without explicit loops over the batch dimension.
    
    Parameters
    -----------
    data : torch.Tensor of shape [n_batch, n, d]
        Vector of data points from which to find neighbors
    queries : torch.Tensor of shape [n_batch, m, d]
        Centers of neighborhoods
    radius : float
        Size of each neighborhood
    
    Returns
    -------
    nbr_dict : dict
        Dictionary containing 'neighbors_index' and 'neighbors_row_splits'
    """
    n_batch, n, d = data.shape
    m = queries.shape[1]

    dists = torch.cdist(queries, data)  # Shape: [n_batch, m, n]
    in_nbr = dists <= radius  # Shape: [n_batch, m, n], bool tensor
    num_neighbors_per_query = in_nbr.sum(dim=2)  # Shape: [n_batch, m]

    max_num_neighbors = num_neighbors_per_query.max().item()
    data_indices = torch.arange(n, device=data.device).view(1, 1, n).expand(n_batch, m, n)
    neighbors_index = torch.full((n_batch, m, max_num_neighbors), -1, device=data.device, dtype=torch.long)

    valid_indices = data_indices * in_nbr.long() + (~in_nbr).long() * (n)
    sorted_indices, _ = torch.sort(valid_indices, dim=2)
    neighbors_index[:, :, :max_num_neighbors] = sorted_indices[:, :, :max_num_neighbors]
    neighbors_index[neighbors_index == n] = -1  # Replace padding indices with -1

    neighbors_row_splits = torch.cat([
        torch.zeros((n_batch, 1), device=data.device, dtype=torch.long),
        num_neighbors_per_query.cumsum(dim=1)
    ], dim=1)  # Shape: [n_batch, m+1]

    nbr_dict = {
        'neighbors_index': neighbors_index,          # Shape: [n_batch, m, max_num_neighbors]
        'neighbors_row_splits': neighbors_row_splits  # Shape: [n_batch, m+1]
    }
    return nbr_dict

def open3d_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """
    Open3D implementation of neighborhood search using tensor operations.
    Supports batched data without explicit loops over the batch dimension.
    
    Parameters
    -----------
    data : torch.Tensor of shape [n_batch, n, d]
        Vector of data points from which to find neighbors
    queries : torch.Tensor of shape [n_batch, m, d]
        Centers of neighborhoods
    radius : float
        Size of each neighborhood

    Returns
    -------
    nbr_dict : dict
        Dictionary containing 'neighbors_index' and 'neighbors_row_splits'
    """
    raise NotImplemented

# Generate sample data
n_batch = 8  # batch size 
n = 9225     
m = 4096       
d = 2        
radius = 0.03

torch.manual_seed(42)


data = torch.rand(n_batch, n, d).to("cuda:0")
queries = torch.rand(n_batch, m, d).to("cuda:0")


neighbor_search_batch = NeighborSearch_batch(use_open3d=False)
result_batched = neighbor_search_batch(data, queries, radius)
breakpoint()