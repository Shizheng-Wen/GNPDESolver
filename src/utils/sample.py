import torch 
from typing import Optional


def subsample(points:torch.Tensor, 
              n:Optional[int] = None, 
              factor:Optional[float] = None)->torch.Tensor:
    """
    Parameters
    ----------
    points: torch.Tensor
        ND tensor
    n: Optional[int]
        number of samples  
    factor: Optional[float]
        factor of subsampling, should be in the range (0, 1]
    Returns
    -------
    sampled_points: torch.Tensor
        1D tensor
    """
    assert n is None or (n > 0 and n <= points.shape[0]), "The number of samples should be in the range (0, n_values]"
    assert factor is None or factor <= 1 and factor > 0, "The factor should be in the range (0, 1]"
    if n is None and factor is None:
        raise ValueError("Either n or factor should be provided")
    if factor is not None:
        n = int(points.shape[0] * factor)
    idx = torch.randperm(points.shape[0])[:n]
    return points[idx]

def grid(points:torch.Tensor, 
            factor:float = 0.5,
            flatten:bool = True)->torch.Tensor:
    """
    Parameters
    ----------
    points: torch.Tensor
        ND tensor of shape [..., n_dimension]
    factor: float
        factor of subsampling, should be in the range (0, 1]
    flatten: bool
        whether to flatten the grid

    Returns
    -------
    sampled_points: torch.Tensor
        if flatten:
            2D tensor of shape [n_points, n_dimension]
        else:
            ND tensor of shape [dim0,dim1,dim2...., n_dimension]
    """
    assert factor <= 1 and factor > 0, "The factor should be in the range (0, 1]"
    points = points.reshape(-1, 2)
    n_points, n_dim = points.shape
    n_sample_per_axis = int((factor * n_points)**(1/n_dim))
    x_min = points.min(0).values
    x_max = points.max(0).values

    _grid = torch.stack(torch.meshgrid(*[torch.linspace(x_min[i].item(), x_max[i].item(), n_sample_per_axis) for i in range(n_dim)]),-1)
    if flatten:
        _grid = _grid.reshape(-1, n_dim)
    return _grid