import os 
import torch
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as np
from tqdm import tqdm

from .base import TrainerBase
from .utils import manual_seed, create_all_to_all_pairs, EPSILON
from architectures.gino import GINO
from data.dataset import Metadata, DATASET_METADATA



class DynamicTrainer(TrainerBase):
    """
    Trainer for dynamic problems, i.e. problems that depend on time.
    """

    def __init__(self, args):
        super().__init__(args)
    
    def init_dataset(self, dataset_config):
        base_path = dataset_config["base_path"]
        dataset_name = dataset_config['name']
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        ds = xr.open_dataset(dataset_path)

        # Load u
        u = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
        u_tensor = torch.tensor(u, dtype=self.dtype)

        # Load c if available
        if self.metadata.group_c is not None:
            c = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            c_tensor = torch.tensor(c, dtype=self.dtype)
        else:
            c_tensor = None

        # Load x
        if self.metadata.group_x is not None:
            x = ds[self.metadata.group_x].values  # Shape: [num_samples, num_timesteps, num_nodes, num_dims]
            x_tensor = torch.tensor(x, dtype=self.dtype)
        else:
            # Generate x coordinates if not available (e.g., for structured grids)
            domain_x = self.metadata.domain_x  # ([xmin, ymin], [xmax, ymax])
            nx, ny = u_tensor.shape[2], u_tensor.shape[3]  # Spatial dimensions
            x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
            y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
            xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')  # [nx, ny]
            x_grid = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
            x_grid = x_grid.reshape(-1, 2)  # [num_nodes, num_dims]
            x_grid = x_grid[None, None, ...]  # Add sample and time dimensions
            x_grid = np.broadcast_to(x_grid, (u_tensor.shape[0], u_tensor.shape[1], x_grid.shape[2], x_grid.shape[3]))  # [num_samples, num_timesteps, num_nodes, num_dims]
            x_tensor = torch.tensor(x_grid, dtype=self.dtype)

        # Handle active variables
        active_vars = self.metadata.active_variables
        u_tensor = u_tensor[..., active_vars]  # Select active variables

        # Compute dataset statistics from training data
        train_size = dataset_config["train_size"]
        val_size = dataset_config["val_size"]
        test_size = dataset_config["test_size"]

        assert train_size + val_size + test_size <= u_tensor.shape[0], "Sum of train, val, and test sizes exceeds total samples"

        # Split data into train, val, test
        train_indices = np.arange(0, train_size)
        val_indices = np.arange(train_size, train_size + val_size)
        test_indices = np.arange(u_tensor.shape[0] - test_size, u_tensor.shape[0])

        u_train = u_tensor[train_indices]
        u_val = u_tensor[val_indices]
        u_test = u_tensor[test_indices]

        x_train = x_tensor[train_indices]
        x_val = x_tensor[val_indices]
        x_test = x_tensor[test_indices]

        if c_tensor is not None:
            c_train = c_tensor[train_indices]
            c_val = c_tensor[val_indices]
            c_test = c_tensor[test_indices]
        else:
            c_train = c_val = c_test = None

        # Compute dataset statistics from training data
        # Reshape u_train to [num_samples * num_timesteps * num_nodes, num_active_vars]
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = u_train_flat.mean(dim=0)
        u_std = u_train_flat.std(dim=0) + EPSILON

        # Store statistics
        self.u_mean = u_mean
        self.u_std = u_std

        # Normalize data
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std

        # If c is used, compute statistics and normalize c
        if c_tensor is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = c_train_flat.mean(dim=0)
            c_std = c_train_flat.std(dim=0) + EPSILON

            # Store statistics
            self.c_mean = c_mean
            self.c_std = c_std

            # Normalize c
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std
        
        # create data pairs for training, validation and test dataset
        u_train_inp, u_train_out, x_train_inp, x_train_out, lead_times_train, time_diffs_train, c_train_inp = create_all_to_all_pairs(u_train, x_train, c_train, self.metadata)
        u_val_inp, u_val_out, x_val_inp, x_val_out, lead_times_val, time_diffs_val, c_val_inp = create_all_to_all_pairs(u_val, x_val, c_val, self.metadata)
        u_test_inp, u_test_out, x_test_inp, x_test_out, lead_times_test, time_diffs_test, c_test_inp = create_all_to_all_pairs(u_test, x_test, c_test, self.metadata)

        


    def init_model(self, model_config):
        pass

    def init_optimizer(self, optimizer_config):
        pass

    def train(self):
        pass    
        