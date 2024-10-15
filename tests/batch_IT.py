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

def native_neighbor_search_(data: torch.Tensor, queries: torch.Tensor, radius: float):
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

@torch.no_grad()
def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float, q_chunk_size=1024):
    n_batch, n, d = data.shape
    m = queries.shape[1]

    neighbors_index_list = []
    num_neighbors_per_query_list = []

    # Change data type to int32 for indices
    data_indices = torch.arange(n, device=data.device, dtype=torch.int32).view(1, 1, n)

    max_num_neighbors = 0

    for start in range(0, m, q_chunk_size):
        end = min(start + q_chunk_size, m)
        queries_chunk = queries[:, start:end, :]  # Shape: [n_batch, q_chunk_size, d]

        # Compute pairwise distances between queries_chunk and data
        dists = torch.cdist(queries_chunk, data)  # Shape: [n_batch, q_chunk_size, n]

        # Determine neighbor relationships
        in_nbr = dists <= radius  # Shape: [n_batch, q_chunk_size, n], bool tensor

        # Count number of neighbors per query point
        num_neighbors_per_query = in_nbr.sum(dim=2)  # Shape: [n_batch, q_chunk_size]
        num_neighbors_per_query_list.append(num_neighbors_per_query)

        max_num_neighbors_chunk = num_neighbors_per_query.max().item()
        max_num_neighbors = max(max_num_neighbors, max_num_neighbors_chunk)

        # Prepare neighbor indices with padding
        data_indices_chunk = data_indices.expand(n_batch, end - start, n)
        valid_indices = torch.where(in_nbr, data_indices_chunk, n)
        sorted_indices, _ = torch.sort(valid_indices, dim=2)
        neighbors_index_chunk = sorted_indices[:, :, :max_num_neighbors_chunk]
        neighbors_index_chunk[neighbors_index_chunk == n] = -1  # Replace padding indices with -1

        # Change indices to int32
        neighbors_index_chunk = neighbors_index_chunk.int()

        # Pad to max_num_neighbors across all chunks
        if neighbors_index_chunk.shape[2] < max_num_neighbors:
            pad_size = max_num_neighbors - neighbors_index_chunk.shape[2]
            neighbors_index_chunk = torch.cat(
                [neighbors_index_chunk, torch.full((n_batch, end - start, pad_size), -1, device=data.device, dtype=torch.int32)],
                dim=2
            )

        neighbors_index_list.append(neighbors_index_chunk)

    # Concatenate results from all chunks
    neighbors_index = torch.cat(neighbors_index_list, dim=1)  # Shape: [n_batch, m, max_num_neighbors]
    num_neighbors_per_query = torch.cat(num_neighbors_per_query_list, dim=1)  # Shape: [n_batch, m]

    # Compute neighbors_row_splits
    neighbors_row_splits = torch.cat([
        torch.zeros((n_batch, 1), device=data.device, dtype=torch.int32),
        num_neighbors_per_query.cumsum(dim=1)
    ], dim=1)  # Shape: [n_batch, m+1]

    nbr_dict = {
        'neighbors_index': neighbors_index,          # Shape: [n_batch, m, max_num_neighbors]
        'neighbors_row_splits': neighbors_row_splits  # Shape: [n_batch, m+1]
    }
    return nbr_dict

##########
# batchIntergralTransform
##########

class IntegralTransformBatch(nn.Module):
    """
    Integral Kernel Transform (GNO) with batched support.
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined
    y : Points over which to integrate
    A(x) : Subset of y points for each x
    k : Kernel parameterized as an MLP

    Parameters
    ----------
    channel_mlp : torch.nn.Module, default None
        MLP parameterizing the kernel k.
    channel_mlp_layers : list, default None
        List of layer sizes for the MLP.
    channel_mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used in the MLP.
    transform_type : str, default 'linear'
        Type of integral transform to compute.
    use_torch_scatter : bool, default True
        Whether to use torch_scatter's segment_csr function.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_torch_scatter=True,
    ):
        super().__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter

        if self.transform_type not in ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"]:
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity
            )
        else:
            self.channel_mlp = channel_mlp

    def forward(self, y, neighbors, x=None, f_y=None, weights=None):
        """Compute a kernel integral transform with batched inputs.

        Parameters
        ----------
        y : torch.Tensor of shape [batch_size, n, d1]
            Points over which to integrate.
        neighbors : dict
            Dictionary containing 'neighbors_index' and 'neighbors_row_splits'.
        x : torch.Tensor of shape [batch_size, m, d2], default None
            Points for which the output is defined. If None, x = y.
        f_y : torch.Tensor of shape [batch_size, n, d3], default None
            Function values at points y.
        weights : torch.Tensor of shape [batch_size, n], default None
            Weights for each point y.

        Returns
        -------
        out_features : torch.Tensor of shape [batch_size, m, d_out]
            Output features at points x.
        """
        if x is None:
            x = y  # x: [batch_size, n, d1]

        batch_size, n, d1 = y.shape
        _, m, d2 = x.shape

        neighbors_index = neighbors["neighbors_index"]  # [batch_size, m, max_num_neighbors]
        neighbors_row_splits = neighbors["neighbors_row_splits"]  # [batch_size, m+1]

        # Replace -1 with 0 in neighbors_index for valid indexing
        neighbors_index_fixed = neighbors_index.clone()
        neighbors_index_fixed[neighbors_index_fixed == -1] = 0  # [batch_size, m, max_num_neighbors]

        # Create mask for valid neighbor indices
        valid_mask = neighbors_index != -1  # [batch_size, m, max_num_neighbors], bool

        # Gather y features at neighbor indices
        y_expanded = y.unsqueeze(1).expand(-1, m, -1, -1)  # [batch_size, m, n, d1]
        rep_features = torch.gather(
            y_expanded,  # [batch_size, m, n, d1]
            2,  # Along the neighbor dimension
            neighbors_index_fixed.unsqueeze(-1).expand(-1, -1, -1, d1)  # [batch_size, m, max_num_neighbors, d1]
        )  # [batch_size, m, max_num_neighbors, d1]

        # Gather f_y features at neighbor indices if f_y is provided
        if f_y is not None:
            _, _, d3 = f_y.shape
            f_y_expanded = f_y.unsqueeze(1).expand(-1, m, -1, -1)  # [batch_size, m, n, d3]
            in_features = torch.gather(
                f_y_expanded,  # [batch_size, m, n, d3]
                2,
                neighbors_index_fixed.unsqueeze(-1).expand(-1, -1, -1, d3)  # [batch_size, m, max_num_neighbors, d3]
            )  # [batch_size, m, max_num_neighbors, d3]
        else:
            in_features = None

        # Prepare self_features from x
        self_features = x.unsqueeze(2).expand(-1, -1, neighbors_index.shape[2], -1)  # [batch_size, m, max_num_neighbors, d2]

        # Concatenate rep_features and self_features
        agg_features = torch.cat([rep_features, self_features], dim=-1)  # [batch_size, m, max_num_neighbors, d1 + d2]

        # Include in_features if required
        if f_y is not None and self.transform_type in ["nonlinear_kernelonly", "nonlinear"]:
            agg_features = torch.cat([agg_features, in_features], dim=-1)  # [batch_size, m, max_num_neighbors, *]

        # Pass through channel MLP
        rep_features = self.channel_mlp(agg_features)  # [batch_size, m, max_num_neighbors, d_out]

        # Apply in_features multiplication if required
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features  # Element-wise multiplication

        # Apply weights if provided
        if weights is not None:
            # Gather weights at neighbor indices
            weights_expanded = weights.unsqueeze(1).expand(-1, m, -1)  # [batch_size, m, n]
            nbr_weights = torch.gather(
                weights_expanded,
                2,
                neighbors_index_fixed
            )  # [batch_size, m, max_num_neighbors]
            nbr_weights = nbr_weights.unsqueeze(-1)  # [batch_size, m, max_num_neighbors, 1]
            rep_features = rep_features * nbr_weights  # Weighted features
            reduction = "sum"
        else:
            reduction = "mean"

        # Zero out invalid neighbor features
        rep_features = rep_features * valid_mask.unsqueeze(-1).float()

        # Sum or mean over neighbors
        if reduction == "sum":
            out_features = rep_features.sum(dim=2)  # [batch_size, m, d_out]
        else:
            # Compute number of valid neighbors per query point
            num_valid_neighbors = valid_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)  # [batch_size, m, 1]
            out_features = rep_features.sum(dim=2) / num_valid_neighbors  # [batch_size, m, d_out]

        return out_features

# Generate sample data
n_batch = 8  # Reduced batch size for demonstration; increase as needed
n = 9225      # Number of data points per batch
m = 4096       # Number of query points per batch
d = 2        # Dimensionality
radius = 0.03

# Seed for reproducibility
torch.manual_seed(42)

# Create random data and queries
data = torch.rand(n_batch, n, d).to("cuda:0")
queries = torch.rand(n_batch, m, d).to("cuda:0")


# Instantiate neighbor search objects
neighbor_search_batch = NeighborSearch_batch(use_open3d=False)
result_batched = neighbor_search_batch(data, queries, radius)
neighbor_search_unbatched = NeighborSearch(use_open3d=False)

# Define the channel MLP layers
input_dim = 2 * d  # For agg_features: concatenation of rep_features and self_features
output_dim = 16    # Must match the dimension of f_y
channel_mlp_layers = [input_dim, 64, output_dim]  # Example MLP layers

# Instantiate the IntegralTransformBatch
integral_transform_batch = IntegralTransformBatch(
    channel_mlp_layers=channel_mlp_layers,
    transform_type="linear",  # Can choose appropriate transform_type
    use_torch_scatter=False
)

# Instantiate the original IntegralTransform
integral_transform_unbatched = IntegralTransform(
    channel_mlp_layers=channel_mlp_layers,
    transform_type="linear",
    use_torch_scatter=False
)

# Copy weights from unbatched to batch version to ensure identical weights
def copy_model_weights(model_src, model_dest):
    src_params = dict(model_src.named_parameters())
    dest_params = dict(model_dest.named_parameters())
    for name in src_params:
        dest_params[name].data.copy_(src_params[name].data)

copy_model_weights(integral_transform_unbatched, integral_transform_batch)

# Generate f_y for all batches
f_y = torch.rand(n_batch, n, output_dim)  # Shape: [n_batch, n, output_dim]

# Now, for each batch, perform unbatched integral transform using neighbor indices from the batched result
outputs_unbatched = []
for b in range(n_batch):
    data_b = data[b]       # Shape: [n, d]
    queries_b = queries[b] # Shape: [m, d]
    
    # Extract neighbor indices and row splits from batched result
    neighbors_index_b = result_batched['neighbors_index'][b]  # Shape: [m, max_num_neighbors]
    neighbors_row_splits_b = result_batched['neighbors_row_splits'][b]  # Shape: [m + 1]
    
    # Flatten neighbors_index_b and remove -1 entries
    neighbors_index_flat = neighbors_index_b.view(-1)
    valid_mask = neighbors_index_flat != -1
    neighbors_index_unbatched = neighbors_index_flat[valid_mask]
    
    # Adjust neighbors_row_splits_b
    # Compute num_neighbors_per_query
    num_neighbors_per_query = (neighbors_index_b != -1).sum(dim=1)
    neighbors_row_splits_unbatched = torch.zeros(m + 1, dtype=torch.long)
    neighbors_row_splits_unbatched[1:] = torch.cumsum(num_neighbors_per_query, dim=0)
    
    neighbors_unbatched = {
        'neighbors_index': neighbors_index_unbatched,         # [num_neighbors_b]
        'neighbors_row_splits': neighbors_row_splits_unbatched  # [m + 1]
    }
    
    # Get f_y for this batch
    f_y_b = f_y[b]  # Shape: [n, output_dim]
    
    # Run the unbatched integral transform
    out_features_unbatched = integral_transform_unbatched(
        y=data_b,
        neighbors=neighbors_unbatched,
        x=queries_b,
        f_y=f_y_b
    )
    
    outputs_unbatched.append(out_features_unbatched)
    
# Run the batched integral transform
out_features_batched = integral_transform_batch(
    y=data,
    neighbors=result_batched,
    x=queries,
    f_y=f_y
)

# Now compare the outputs
for b in range(n_batch):
    out_unbatched = outputs_unbatched[b]  # Shape: [m, output_dim]
    out_batched = out_features_batched[b]  # Shape: [m, output_dim]
    
    # Compare outputs
    if not torch.allclose(out_unbatched, out_batched, atol=1e-6):
        print(f"Batch {b} outputs are not equal.")
    else:
        print(f"Batch {b} outputs are equal.")

