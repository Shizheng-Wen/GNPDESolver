import torch
import torch.nn as nn
import sys
sys.path.append("../")
from src.model.cmpt.utils.gno_utils import NeighborSearch

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

    # Compute pairwise distances between queries and data
    dists = torch.cdist(queries, data)  # Shape: [n_batch, m, n]

    # Determine neighbor relationships
    in_nbr = dists <= radius  # Shape: [n_batch, m, n], bool tensor

    # Count number of neighbors per query point
    num_neighbors_per_query = in_nbr.sum(dim=2)  # Shape: [n_batch, m]

    # Compute maximum number of neighbors per query across the entire batch
    max_num_neighbors = num_neighbors_per_query.max().item()

    # Prepare neighbor indices with padding
    data_indices = torch.arange(n, device=data.device).view(1, 1, n).expand(n_batch, m, n)
    neighbors_index = torch.full((n_batch, m, max_num_neighbors), -1, device=data.device, dtype=torch.long)

    # Mask data indices where in_nbr is False
    valid_indices = data_indices * in_nbr.long() + (~in_nbr).long() * (n)
    sorted_indices, _ = torch.sort(valid_indices, dim=2)
    neighbors_index[:, :, :max_num_neighbors] = sorted_indices[:, :, :max_num_neighbors]
    neighbors_index[neighbors_index == n] = -1  # Replace padding indices with -1

    # Compute neighbors_row_splits
    # Since we have padded neighbors_index, row_splits can be derived from num_neighbors_per_query
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
    from open3d.ml.torch.layers import FixedRadiusSearch

    n_batch, n, d = data.shape
    m = queries.shape[1]

    # Flatten data and queries
    data_flat = data.reshape(-1, d)         # Shape: [n_batch * n, d]
    queries_flat = queries.reshape(-1, d)   # Shape: [n_batch * m, d]

    # Construct row splits for data and queries
    batch_indices = torch.arange(n_batch + 1, device=data.device) * n
    points_row_splits = batch_indices.long()
    queries_row_splits = (torch.arange(n_batch + 1, device=queries.device) * m).long()

    # Call Open3D's FixedRadiusSearch
    frs = FixedRadiusSearch()
    search_return = frs(
        data_flat, queries_flat, radius,
        points_row_splits=points_row_splits,
        queries_row_splits=queries_row_splits
    )

    # Extract neighbors_index and neighbors_row_splits
    neighbors_index_flat = search_return.neighbors_index.long()  # Shape: [num_neighbors_total]
    neighbors_row_splits_flat = search_return.neighbors_row_splits.long()  # Shape: [n_batch * m + 1]

    # Compute number of neighbors per query
    num_neighbors_per_query = neighbors_row_splits_flat[1:] - neighbors_row_splits_flat[:-1]  # Shape: [n_batch * m]
    max_num_neighbors = num_neighbors_per_query.max().item()

    # Prepare neighbors_index tensor
    neighbors_index = torch.full((n_batch * m, max_num_neighbors), -1, device=data.device, dtype=torch.long)
    idx = torch.arange(neighbors_index_flat.shape[0], device=data.device)

    # Compute per-query indices
    query_indices = torch.repeat_interleave(torch.arange(n_batch * m, device=data.device), num_neighbors_per_query)
    neighbor_positions = (neighbors_row_splits_flat[:-1].unsqueeze(1) + torch.arange(max_num_neighbors, device=data.device).unsqueeze(0)).flatten()

    valid_mask = idx < neighbors_row_splits_flat[1:].repeat_interleave(max_num_neighbors)
    neighbors_index_flat_padded = torch.full((n_batch * m * max_num_neighbors,), -1, device=data.device, dtype=torch.long)
    neighbors_index_flat_padded[valid_mask] = neighbors_index_flat
    neighbors_index = neighbors_index_flat_padded.view(n_batch * m, max_num_neighbors)

    # Reshape neighbors_index and neighbors_row_splits to include batch dimension
    neighbors_index = neighbors_index.view(n_batch, m, max_num_neighbors)
    neighbors_row_splits = torch.cat([
        torch.zeros((n_batch, 1), device=data.device, dtype=torch.long),
        num_neighbors_per_query.view(n_batch, m).cumsum(dim=1)
    ], dim=1)  # Shape: [n_batch, m+1]

    nbr_dict = {
        'neighbors_index': neighbors_index,          # Shape: [n_batch, m, max_num_neighbors]
        'neighbors_row_splits': neighbors_row_splits  # Shape: [n_batch, m+1]
    }
    return nbr_dict

n_batch = 36  # Number of batches
n = 500       # Number of data points per batch
m = 400       # Number of query points per batch
d = 3         # Dimensionality
radius = 0.5

# Seed for reproducibility
torch.manual_seed(42)

# Create random data and queries
data = torch.rand(n_batch, n, d)
queries = torch.rand(n_batch, m, d)

# Instantiate the batched neighbor search class
neighbor_search_batch = NeighborSearch_batch(use_open3d=False)
result_batched = neighbor_search_batch(data, queries, radius)

# Instantiate the original unbatched neighbor search class
neighbor_search_unbatched = NeighborSearch(use_open3d=False)

# List to store unbatched results
results_unbatched = []

# Process each batch individually using the unbatched neighbor search
for b in range(n_batch):
    data_b = data[b]       # Shape: [n, d]
    queries_b = queries[b] # Shape: [m, d]

    # Perform neighbor search for this batch
    result_unbatched = neighbor_search_unbatched(data_b, queries_b, radius)
    results_unbatched.append(result_unbatched)

# Process each batch individually using the unbatched neighbor search
for b in range(n_batch):
    data_b = data[b]       # Shape: [n, d]
    queries_b = queries[b] # Shape: [m, d]

    # Perform neighbor search for this batch
    result_unbatched = neighbor_search_unbatched(data_b, queries_b, radius)
    results_unbatched.append(result_unbatched)

# Compare the results between batched and unbatched methods
for b in range(n_batch):
    print(f"\nBatch {b}:")
    neighbors_index_batched = result_batched['neighbors_index'][b]  # Shape: [m, max_num_neighbors]
    neighbors_row_splits_batched = result_batched['neighbors_row_splits'][b]  # Shape: [m+1]

    neighbors_index_unbatched = results_unbatched[b]['neighbors_index']  # Shape: [num_neighbors_in_batch]
    neighbors_row_splits_unbatched = results_unbatched[b]['neighbors_row_splits']  # Shape: [m+1]

    # Reconstruct neighbors_index from batched output by removing padding (-1)
    reconstructed_indices = []
    for i in range(m):
        num_neighbors = neighbors_row_splits_batched[i + 1] - neighbors_row_splits_batched[i]
        indices = neighbors_index_batched[i, :num_neighbors]
        reconstructed_indices.append(indices)
    neighbors_index_batched_flat = torch.cat(reconstructed_indices, dim=0)

    # Check if the neighbor indices are the same
    indices_equal = torch.equal(
        torch.sort(neighbors_index_batched_flat)[0],
        torch.sort(neighbors_index_unbatched)[0]
    )
    print(f"  Neighbors indices equal: {indices_equal}")

    # Check if the row splits are the same
    row_splits_equal = torch.equal(
        neighbors_row_splits_batched,
        neighbors_row_splits_unbatched
    )
    print(f"  Neighbors row splits equal: {row_splits_equal}")

    # Detailed output (optional)
    if not indices_equal or not row_splits_equal:
        print("Detailed comparison:")
        print("Batched neighbors_index:", neighbors_index_batched_flat)
        print("Unbatched neighbors_index:", neighbors_index_unbatched)
        print("Batched neighbors_row_splits:", neighbors_row_splits_batched)
        print("Unbatched neighbors_row_splits:", neighbors_row_splits_unbatched)