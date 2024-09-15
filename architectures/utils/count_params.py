from torch import prod

def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )

def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2*n_params
    return n_params

