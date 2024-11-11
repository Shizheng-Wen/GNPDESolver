import torch
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import os


def rescale(x:torch.Tensor, lims=(-1,1))->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor
    
    Returns
    -------
    x_normalized: torch.Tensor
        ND tensor
    """
    return (x-x.min()) / (x.max()-x.min()) * (lims[1] - lims[0]) + lims[0]

def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """
    Native PyTorch implementation of a neighborhood search
    between two arbitrary coordinate meshes.
     
    Parameters
    -----------

    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """

    # compute pairwise distances
    dists = torch.cdist(queries, data).to(queries.device) # shaped num query points x num data points
    in_nbr = torch.where(dists <= radius, 1., 0.) # i,j is one if j is i's neighbor
    nbr_indices = in_nbr.nonzero()[:,1:].reshape(-1,) # only keep the column indices
    nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0) # num points in each neighborhood, summed cumulatively
    splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
    nbr_dict['neighbors_row_splits'] = splits.long()
    return nbr_dict

def get_neighbor_counts(nbr_dict: dict) -> torch.Tensor:
    """
    Count the number of neighbor for each query.
    
    Parameters
    ----------
    nbr_dict : dict
        Dic return from native_neighbor_search.
    
    Returns
    -------
    torch.Tensor
        The number of neighbor points for each queries.
    """
    splits = nbr_dict['neighbors_row_splits']
    neighbor_counts = splits[1:] - splits[:-1]
    return neighbor_counts

def plot_neighbor_distribution(neighbor_counts: torch.Tensor, num_queries: int):
    """
    Plot the histogram of neighbor distribution.
    
    Parameters
    ----------
    neighbor_counts : torch.Tensor
        The number of neighbors for each query.
    num_queries : int
        The number of total queries.
    """

    neighbor_counts = neighbor_counts.cpu().numpy()
    

    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    sns.histplot(neighbor_counts, bins=30, kde=False, color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Neighbor Counts per Query Point', fontsize=16)
    plt.xlabel('Number of Neighbors', fontsize=14)
    plt.ylabel('Number of Query Points', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("neighbor_num_dist.png")

def count_queries_without_neighbors(neighbor_counts: torch.Tensor) -> int:
    """
    Count the neighbors that doesn't have the neighbors.
    
    Parameters
    ----------
    neighbor_counts : torch.Tensor
        The number of points for each query.
    
    Returns
    -------
    int
        The number of queries that don't have the neighbors.
    """
    no_neighbor_count = torch.sum(neighbor_counts == 0).item()
    return no_neighbor_count

if __name__ == "__main__":
    torch.manual_seed(42)
    
    base_path = "/cluster/work/math/camlab-data/rigno-unstructured/"
    dataset_name = "CE-Gauss"
    dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
    with xr.open_dataset(dataset_path) as ds:
        x_array = ds['x'].values[0][0]

    poseidon_dataset_name = ["CE-RP", "CE-Gauss","NS-PwC",
                                        "NS-SVS","NS-Gauss","NS-SL",
                                        "ACE", "Wave-Layer"]
    if dataset_name in poseidon_dataset_name:
        x_array = x_array[:9216,:]
    
    
    physical_points = rescale(torch.tensor(x_array), (-1,1))

    meshgrid = torch.meshgrid(torch.linspace(0, 1, 64),
                              torch.linspace(0, 1, 64),
                              indexing='ij')
    regional_points = rescale(torch.stack(meshgrid, dim=-1).reshape(-1,2), (-1,1)).to(physical_points.dtype)
    num_queries = regional_points.shape[0]
    radius = 0.055
    nbr_dict = native_neighbor_search(physical_points, regional_points, radius)
    neighbor_counts = get_neighbor_counts(nbr_dict)
    no_neighbors = count_queries_without_neighbors(neighbor_counts)
    print(f"The queries without the neighbors: {no_neighbors} out of {num_queries}")
    plot_neighbor_distribution(neighbor_counts, num_queries)