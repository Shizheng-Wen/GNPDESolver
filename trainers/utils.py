import numpy as np
import torch
import torch.nn as nn
from data.dataset import Metadata

EPSILON = 1e-10

def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def init_random_seed():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1)
    np.random.seed(1)

def save_ckpt(path, **kwargs):
    """
        Save checkpoint to the path

        Usage:
        >>> save_ckpt("model/poisson_1000.pt", model=model, optimizer=optimizer, scheduler=scheduler)

        Parameters:
        -----------
            path: str
                path to save the checkpoint
            kwargs: dict
                key: str
                    name of the model
                value: StateFul torch object which has the .state_dict() method
                    save object
        
    """
    for k, v in kwargs.items():
        kwargs[k] = v.state_dict()
    torch.save(kwargs, path)

def load_ckpt(path, **kwargs):
    """
        Load checkpoint from the path

        Usage:
        >>> model, optimizer, scheduler = load_ckpt("model/poisson_1000.pt", model=model, optimizer=optimizer, scheduler=scheduler)

        Parameters:
        -----------
            path: str
                path to load the checkpoint
            kwargs: dict
                key: str
                    name of the model
                value: StateFul torch object which has the .state_dict() method
                    save object
        Returns:
        --------
            list of torch object
            [model, optimizer, scheduler]
    """
    ckpt = torch.load(path)
    for k, v in kwargs.items():
        kwargs[k].load_state_dict(ckpt[k])
    return [i for i in kwargs.values()]

def compute_batch_errors(gtr: torch.Tensor, prd: torch.Tensor, metadata: Metadata) -> torch.Tensor:
    """
    Compute the per-sample relative L1 errors per variable chunk for a batch.
    
    Args:
        gtr (torch.Tensor): Ground truth tensor with shape [batch_size, time, space, var]
        prd (torch.Tensor): Predicted tensor with shape [batch_size, time, space, var]
        metadata (Metadata): Dataset metadata including global_mean, global_std, and variable chunks
    
    Returns:
        torch.Tensor: Relative errors per sample per variable chunk, shape [batch_size, num_chunks]
    """
    # normalize the data
    active_vars = metadata.active_variables

    mean = torch.tensor(metadata.global_mean, device=gtr.device, dtype=gtr.dtype)[active_vars].reshape(1, 1, 1, -1)
    std = torch.tensor(metadata.global_std, device=gtr.device, dtype=gtr.dtype)[active_vars].reshape(1, 1, 1, -1)
    
    original_chunks = metadata.chunked_variables
    chunked_vars = [original_chunks[i] for i in active_vars]
    unique_chunks = sorted(set(chunked_vars))
    chunk_map = {old_chunk: new_chunk for new_chunk, old_chunk in enumerate(unique_chunks)}
    adjusted_chunks = [chunk_map[chunk] for chunk in chunked_vars]
    num_chunks = len(unique_chunks)

    chunks = torch.tensor(adjusted_chunks, device=gtr.device, dtype=torch.long)  # Shape: [var]

    gtr_norm = (gtr - mean) / std
    prd_norm = (prd - mean) / std

    # compute absolute errors and sum over the time and space dimensions
    abs_error = torch.abs(gtr_norm - prd_norm)  # Shape: [batch_size, time, space, var]
    error_sum = torch.sum(abs_error, dim=(1, 2))  # Shape: [batch_size, var]

    # sum errors per variable chunk
    # chunks = torch.tensor(metadata.chunked_variables, device = gtr.device, dtype = torch.long) # Shape: [var]
    # num_chunks = metadata.num_variable_chunks
    chunks_expanded = chunks.unsqueeze(0).expand(error_sum.size(0), -1)  # Shape: [batch_size, var]
    error_per_chunk = torch.zeros(error_sum.size(0), num_chunks, device=gtr.device, dtype=error_sum.dtype)
    error_per_chunk.scatter_add_(1, chunks_expanded, error_sum)

    # compute sum of absolute values of the ground truth per chunk
    gtr_abs_sum = torch.sum(torch.abs(gtr_norm), dim=(1, 2))  # Shape: [batch_size, var]
    gtr_sum_per_chunk = torch.zeros(gtr_abs_sum.size(0), num_chunks, device=gtr.device, dtype=gtr_abs_sum.dtype)
    gtr_sum_per_chunk.scatter_add_(1, chunks_expanded, gtr_abs_sum)

    # compute relative errors per chunk
    relative_error_per_chunk = error_per_chunk / (gtr_sum_per_chunk + EPSILON) # Shape: [batch_size, num_chunks]

    return relative_error_per_chunk # Shape: [batch_size, num_chunks]
    
def compute_final_metric(all_relative_errors: torch.Tensor) -> float:
    """
    Compute the final metric from the accumulated relative errors.
    
    Args:
        all_relative_errors (torch.Tensor): Tensor of shape [num_samples, num_chunks]
        
    Returns:
        Metrics: An object containing the final relative L1 median error
    """
    # Step 3: Compute the median over the sample axis for each chunk
    median_error_per_chunk = torch.median(all_relative_errors, dim=0)[0]  # Shape: [num_chunks]

    # Step 4: Take the mean of the median errors across all chunks
    final_metric = torch.mean(median_error_per_chunk)
    
    return final_metric.item()

def create_all_to_all_pairs(u_data: torch.Tensor , x_data: torch.Tensor, c_data: torch.Tensor, metadata: Metadata):
    """
        This code is mainly used for time-dependent PDE, for constructing the all-to-all pairs dataset.
    """
    num_samples, num_timesteps, num_nodes, num_vars = u_data.shape
    device = u_data.device
    dtype = u_data.dtype

    if metadata.domain_t is not None:
        t_start, t_end = metadata.domain_t  # For example, (0.0, 1.0)
    else:
        raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")

    # Compute actual time values
    time_values = torch.linspace(t_start, t_end, num_timesteps, device=device, dtype=dtype)  # Shape: [num_timesteps]

    # Create all possible combinations of t_in and t_out where t_out > t_in
    t_in_indices, t_out_indices = torch.meshgrid(torch.arange(num_timesteps, device=device), torch.arange(num_timesteps, device=device), indexing='ij')
    t_in_indices = t_in_indices.flatten()  # Shape: [num_timesteps^2]
    t_out_indices = t_out_indices.flatten()

    # Only keep pairs where t_out > t_in, but it can be optimized with max_time_dff
    # for example:
    # max_time_diff = 5
    # time_diffs = t_out_indices - t_in_indices
    # mask = (t_out_indices > t_in_indices) & (time_diffs <= max_time_diff)
    mask = t_out_indices > t_in_indices
    t_in_indices = t_in_indices[mask]
    t_out_indices = t_out_indices[mask]

    num_pairs = t_in_indices.shape[0]
    assert num_timesteps * (num_timesteps - 1) // 2 == num_pairs, f"Expected {num_timesteps * (num_timesteps - 1) // 2} pairs for every data, but found {num_pairs}."

    # Expand sample indices
    sample_indices = torch.arange(num_samples, device=device).unsqueeze(1).repeat(1, num_pairs).flatten()  # Shape: [num_samples * num_pairs]

    # Repeat t_in_indices and t_out_indices for all samples
    t_in_indices_expanded = t_in_indices.unsqueeze(0).repeat(num_samples, 1).flatten()  # Shape: [num_samples * num_pairs]
    t_out_indices_expanded = t_out_indices.unsqueeze(0).repeat(num_samples, 1).flatten()

    # Get u_in and u_out
    u_in = u_data[sample_indices, t_in_indices_expanded, :, :]  # Shape: [num_samples * num_pairs, num_nodes, num_vars]
    u_out = u_data[sample_indices, t_out_indices_expanded, :, :]  # Same shape

    # Get x_in and x_out
    x_in = x_data[sample_indices, t_in_indices_expanded, :, :]  # Shape: [num_samples * num_pairs, num_nodes, num_dims]
    x_out = x_data[sample_indices, t_out_indices_expanded, :, :]  # Same shape

    # Map time indices to actual time values
    t_in_times = time_values[t_in_indices_expanded]  # Shape: [num_samples * num_pairs]
    t_out_times = time_values[t_out_indices_expanded]  # Same shape

    # Compute lead_times and time_diffs in actual time units
    lead_times = t_out_times.unsqueeze(1)  # Shape: [num_samples * num_pairs, 1]
    time_diffs = (t_out_times - t_in_times).unsqueeze(1)  # Shape: [num_samples * num_pairs, 1]

    if c_data is not None:
        c_in = c_data[sample_indices, t_in_indices_expanded, :, :]  # Shape: [num_samples * num_pairs, num_nodes, num_c_vars]
    else:
        c_in = None

    return u_in, u_out, x_in, x_out, lead_times, time_diffs, c_in