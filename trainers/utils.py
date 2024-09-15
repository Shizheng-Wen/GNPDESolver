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
