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

EPSILON = 1e-10

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

        with xr.open_dataset(dataset_path) as ds:
            # Load u as NumPy array
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]

            # Load c if available
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None

            # Load x
            if self.metadata.group_x is not None:
                x_array = ds[self.metadata.group_x].values  # Shape: [1, 1, num_nodes, num_dims]
                x_array = np.broadcast_to(x_array, (u_array.shape[0], u_array.shape[1], x_array.shape[2], x_array.shape[3]))  # Shape: [num_samples, num_timesteps, num_nodes, num_dims]
            else:
                # Generate x coordinates if not available (e.g., for structured grids)
                domain_x = self.metadata.domain_x  # ([xmin, ymin], [xmax, ymax])
                nx, ny = u_array.shape[2], u_array.shape[3]  # Spatial dimensions
                x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')  # [nx, ny]
                x_grid = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
                x_grid = x_grid.reshape(-1, 2)  # [num_nodes, num_dims]
                x_grid = x_grid[None, None, ...]  # Add sample and time dimensions
                x_array = np.broadcast_to(x_grid, (u_array.shape[0], u_array.shape[1], x_grid.shape[2], x_grid.shape[3]))  # [num_samples, num_timesteps, num_nodes, num_dims]

        # Handle active variables
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]  # Select active variables
        # Or, if you prefer to select specific variables:
        # u_array = u_array[:, :, :, 0:2]
    
        # Compute dataset sizes
        train_size = dataset_config["train_size"]
        val_size = dataset_config["val_size"]
        test_size = dataset_config["test_size"]

        assert train_size + val_size + test_size <= u_array.shape[0], "Sum of train, val, and test sizes exceeds total samples"

        # Split data into train, val, test
        u_train = u_array[:train_size]
        u_val = u_array[train_size:train_size+val_size]
        u_test = u_array[-test_size:]

        x_train = x_array[:train_size]
        x_val = x_array[train_size:train_size+val_size]
        x_test = x_array[-test_size:]

        if c_array is not None:
            c_train = c_array[:train_size]
            c_val = c_array[train_size:train_size+val_size]
            c_test = c_array[-test_size:]
        else:
            c_train = c_val = c_test = None

        # Compute dataset statistics from training data
        # Reshape u_train to [num_samples * num_timesteps * num_nodes, num_active_vars]
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + 1e-8  # Avoid division by zero

        # Store statistics as torch tensors
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)

        # Normalize data using NumPy operations
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std

        # If c is used, compute statistics and normalize c
        if c_array is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + 1e-8  # Avoid division by zero

            # Store statistics
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)

            # Normalize c
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std

        # Create data pairs for training, validation, and test datasets
        u_train_inp, u_train_out, x_train_inp, x_train_out, lead_times_train, time_diffs_train, c_train_inp = create_all_to_all_pairs(u_train, x_train, c_train, self.metadata)
        u_val_inp, u_val_out, x_val_inp, x_val_out, lead_times_val, time_diffs_val, c_val_inp = create_all_to_all_pairs(u_val, x_val, c_val, self.metadata)
        u_test_inp, u_test_out, x_test_inp, x_test_out, lead_times_test, time_diffs_test, c_test_inp = create_all_to_all_pairs(u_test, x_test, c_test, self.metadata)
        # Optionally normalize lead_times and time_diffs
        time_diffs_mean = np.mean(time_diffs_train)
        time_diffs_std = np.std(time_diffs_train) + EPSILON
        lead_times_mean = np.mean(lead_times_train)
        lead_times_std = np.std(lead_times_train) + EPSILON

        # Store time statistics
        self.time_stats = {
            'lead_times_mean': torch.tensor(lead_times_mean, dtype=self.dtype),
            'lead_times_std': torch.tensor(lead_times_std, dtype=self.dtype),
            'time_diffs_mean': torch.tensor(time_diffs_mean, dtype=self.dtype),
            'time_diffs_std': torch.tensor(time_diffs_std, dtype=self.dtype)
        }

        # Normalize time features
        lead_times_train = (lead_times_train - lead_times_mean) / lead_times_std
        time_diffs_train = (time_diffs_train - time_diffs_mean) / time_diffs_std

        lead_times_val = (lead_times_val - lead_times_mean) / lead_times_std
        time_diffs_val = (time_diffs_val - time_diffs_mean) / time_diffs_std

        lead_times_test = (lead_times_test - lead_times_mean) / time_diffs_std
        time_diffs_test = (time_diffs_test - time_diffs_mean) / time_diffs_std

        # Convert data to PyTorch tensors only when creating TensorDataset
        if c_array is not None:
            # Stack inputs
            train_inputs = np.concatenate([u_train_inp, c_train_inp, x_train_inp], axis=-1)  # Shape: [num_samples, num_nodes, input_dim]
            val_inputs = np.concatenate([u_val_inp, c_val_inp, x_val_inp], axis=-1)
            test_inputs = np.concatenate([u_test_inp, c_test_inp, x_test_inp], axis=-1)
        else:
            train_inputs = np.concatenate([u_train_inp, x_train_inp], axis=-1)
            val_inputs = np.concatenate([u_val_inp, x_val_inp], axis=-1)
            test_inputs = np.concatenate([u_test_inp, x_test_inp], axis=-1)

        # Add lead_times and time_diffs
        # Since lead_times and time_diffs are of shape [num_samples, 1], we need to expand them to match the spatial dimension
        num_nodes = train_inputs.shape[1]
        lead_times_train_expanded = np.repeat(lead_times_train[:, np.newaxis, :], num_nodes, axis=1)
        time_diffs_train_expanded = np.repeat(time_diffs_train[:, np.newaxis, :], num_nodes, axis=1)

        lead_times_val_expanded = np.repeat(lead_times_val[:, np.newaxis, :], num_nodes, axis=1)
        time_diffs_val_expanded = np.repeat(time_diffs_val[:, np.newaxis, :], num_nodes, axis=1)

        lead_times_test_expanded = np.repeat(lead_times_test[:, np.newaxis, :], num_nodes, axis=1)
        time_diffs_test_expanded = np.repeat(time_diffs_test[:, np.newaxis, :], num_nodes, axis=1)

        # Concatenate time features
        train_inputs = np.concatenate([train_inputs, lead_times_train_expanded, time_diffs_train_expanded], axis=-1)
        val_inputs = np.concatenate([val_inputs, lead_times_val_expanded, time_diffs_val_expanded], axis=-1)
        test_inputs = np.concatenate([test_inputs, lead_times_test_expanded, time_diffs_test_expanded], axis=-1)

        # Convert to PyTorch tensors
        train_inputs = torch.tensor(train_inputs, dtype=self.dtype)
        val_inputs = torch.tensor(val_inputs, dtype=self.dtype)
        test_inputs = torch.tensor(test_inputs, dtype=self.dtype)

        u_train_out = torch.tensor(u_train_out, dtype=self.dtype)
        u_val_out = torch.tensor(u_val_out, dtype=self.dtype)
        u_test_out = torch.tensor(u_test_out, dtype=self.dtype)

        # Create TensorDatasets
        train_ds = TensorDataset(train_inputs, u_train_out)
        val_ds = TensorDataset(val_inputs, u_val_out)
        test_ds = TensorDataset(test_inputs, u_test_out)

        # Create DataLoaders
        self.train_loader = DataLoader(train_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.val_loader = DataLoader(val_ds, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])
        self.test_loader = DataLoader(test_ds, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])

        # Generate latent queries (if needed)
        x_min, y_min = self.metadata.domain_x[0]
        x_max, y_max = self.metadata.domain_x[1]
        meshgrid = np.meshgrid(np.linspace(x_min, x_max, dataset_config["latent_queries"][0]),
                            np.linspace(y_min, y_max, dataset_config["latent_queries"][1]),
                            indexing='ij')
        latent_queries = np.stack(meshgrid, axis=-1).reshape(-1, 2)  # [num_queries, 2]
        self.latent_queries = torch.tensor(latent_queries, dtype=self.dtype).unsqueeze(0).to(self.device)  # [1, num_queries, 2]

    def init_model(self, model_config):
        in_channels = self.u_mean.shape[0] + 2

        if hasattr(self, 'c_mean'):
            in_channels += self.c_mean.shape[0]

        out_channels = self.u_mean.shape[0]

        self.model = GINO(in_channels=in_channels, out_channels=out_channels, **model_config["args"])

    def train(self):
        self.model.train()
        self.model.to(self.device)

        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample, coord_sample) in enumerate(self.test_loader):
                x_sample, y_sample, coord_sample = x_sample.to(self.device), y_sample.to(self.device), coord_sample.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                pred = self.model(x=x_sample.squeeze(1), input_geom=coord_sample.squeeze(1)[0:1], latent_queries=self.latent_queries, output_queries=coord_sample.squeeze(1)[0]).unsqueeze(1) # Shape: [batch_size, 1, num_nodes, num_channels]
                
                # TODO: check the correctness of the next line
                pred_de_norm = pred * self.u_std + self.u_mean
                y_sample_de_norm = y_sample * self.u_std + self.u_mean
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (poseidon_metric)"] = final_metric
        # TODO: finish the plot_animation function

    def plot_animation(self, coords, input, gt, pred):
        pass

        