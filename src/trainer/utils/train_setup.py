import numpy as np
import torch
import random

def manual_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
