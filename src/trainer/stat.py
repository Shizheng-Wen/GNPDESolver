import os 
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from .base import TrainerBase
from .utils import manual_seed, compute_batch_errors, compute_final_metric
from .utils.plot import plot_estimates
from ..utils import shallow_asdict

from src.data.dataset import Metadata, DATASET_METADATA
from src.graph import RegionInteractionGraph
from src.model import init_model_from_rigraph

EPSILON = 1e-10

class StaticTrainer(TrainerBase):
    """
    Trainer for static problems, i.e. problems that do not depend on time.
    """

    def __init__(self, args):
        super().__init__(args)
   
    def init_dataset(self, dataset_config):
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        self.poseidon_dataset_name = ["Poisson-Gauss"]
        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            if self.metadata.group_x is not None:
                x_array = ds[self.metadata.group_x].values
                self.all_coord = x_array
                if x_array.shape[0] == u_array.shape[0]:
                   x_array = x_array[0:1] # TODO: x_array is not constant across samples dimension.
                self.x_train = x_array
            else:
                domain_x = self.metadata.domain_x #([xmin, ymin], [xmax, ymax])
                nx, ny = u_array.shape[-2], u_array.shape[-1]
                x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
                x_grid = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
                x_grid = x_grid.reshape(-1, 2)  # [num_nodes, num_dims]
                x_grid = x_grid[None, None, ...]  # Add sample and time dimensions
                self.x_train = x_grid 
                self.all_coord = x_grid
                c_array = c_array.reshape(c_array.shape[0],-1)[:,None,:,None] # [num_samples, 1, num_nodes, 1]
                u_array = u_array.reshape(u_array.shape[0],-1)[:,None,:,None] # [num_samples, 1, num_nodes, 1]
                

        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            self.x_train = self.x_train[:,:,:9216,:]
       
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = c_array.shape[-1]
        self.num_output_channels = u_array.shape[-1]

        # Compute dataset sizes
        total_samples = u_array.shape[0]
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."

        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = u_array[train_indices]
        u_val = u_array[val_indices]
        u_test = u_array[test_indices]

        if c_array is not None:
            c_train = c_array[train_indices]
            c_val = c_array[val_indices]
            c_test = c_array[test_indices]
        else:
            c_train = c_val = c_test = None
        
        # Compute dataset statistics from training data
        # Reshape u_train to [num_samples * num_timesteps * num_nodes, num_active_vars]
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON  # Avoid division by zero
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)
        # Normalize the data
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std
        # If c is used, compute statistics and normalize c
        if c_array is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + EPSILON 
            # Store statistics
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)
            # Normalize c
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std

        self.x_train = torch.tensor(self.x_train, dtype=self.dtype)#.to(self.device) # self.x_train is numpy.array before
        
        c_train, u_train = torch.tensor(c_train, dtype=self.dtype), torch.tensor(u_train, dtype=self.dtype)
        c_val, u_val = torch.tensor(c_val, dtype=self.dtype), torch.tensor(u_val, dtype=self.dtype)
        c_test, u_test = torch.tensor(c_test, dtype=self.dtype), torch.tensor(u_test, dtype=self.dtype)
        train_ds = TensorDataset(c_train, u_train)
        val_ds = TensorDataset(c_val, u_val)
        test_ds = TensorDataset(c_test, u_test)

        self.train_loader = DataLoader(train_ds, batch_size=dataset_config.batch_size, shuffle=dataset_config.shuffle, num_workers=dataset_config.num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=dataset_config.batch_size, shuffle=dataset_config.shuffle, num_workers=dataset_config.num_workers)
        self.test_loader = DataLoader(test_ds, batch_size=dataset_config.batch_size, shuffle=dataset_config.shuffle, num_workers=dataset_config.num_workers)

    def init_graph(self, graph_config):
        self.rigraph = RegionInteractionGraph.from_point_cloud(points = self.x_train[0][0],
                                              phy_domain=self.metadata.domain_x,
                                              **shallow_asdict(graph_config)
                                            )
        # record the number of edges
        self.config.datarow['p2r edges'] = self.rigraph.physical_to_regional.num_edges
        self.config.datarow['r2r edges'] = self.rigraph.regional_to_regional.num_edges
        self.config.datarow['r2p edges'] = self.rigraph.regional_to_physical.num_edges
                                                    
    def init_model(self, model_config):
        self.model = init_model_from_rigraph(rigraph=self.rigraph, 
                                            input_size=self.num_input_channels, 
                                            output_size=self.num_output_channels, 
                                            model=model_config.name,
                                            drop_edge=model_config.drop_edge,
                                            variable_mesh=model_config.variable_mesh,
                                            config=model_config.args
                                            )
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample) in enumerate(self.test_loader):
                x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                x_sample, y_sample = x_sample.squeeze(1), y_sample.squeeze(1)
                pred = self.model(self.rigraph, x_sample)
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_sample_de_norm = y_sample * self.u_std.to(self.device) + self.u_mean.to(self.device)
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (direct)"] = final_metric
        print(f"relative error: {final_metric}")

        x_plot = x_sample * self.c_std.to(self.device)  

        fig = plot_estimates(
            u_inp = x_sample[-1].cpu().numpy(), 
            u_gtr = y_sample_de_norm[-1].cpu().numpy(), 
            u_prd = pred_de_norm[-1].cpu().numpy(), 
            x_inp = self.all_coord[-1][0],
            x_out = self.all_coord[-1][0],
            names = self.metadata.names['u'],
            symmetric = self.metadata.signed['u'],
            domain = self.metadata.domain_x)

        fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    def plot_results(self, coords, input, gt, pred):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        im1 = ax1.tricontourf(coords[:, 0].cpu(), coords[:, 1].cpu(), input[:, 0].cpu(), cmap='plasma')
        ax1.set_title("Input (c)")
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.tricontourf(coords[:, 0].cpu(), coords[:, 1].cpu(), gt[:, 0].cpu(), cmap='plasma')
        ax2.set_title("Ground Truth (u)")
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.tricontourf(coords[:, 0].cpu(), coords[:, 1].cpu(), pred[:, 0].cpu(), cmap='plasma')
        ax3.set_title("Prediction")
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.savefig(self.path_config.result_path)
        plt.close()
        
    def measure_inference_time(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            # Get a single sample from the test dataset
            x_sample, y_sample = self.test_loader.dataset[0]
            # Ensure inputs are tensors and add batch dimension
            x_sample = x_sample.to(self.device).unsqueeze(0)  # Shape: [1, num_timesteps, num_nodes, num_channels]
            # Since it's a static problem, squeeze the time dimension
            x_input = x_sample.squeeze(1)  # Shape: [1, num_nodes, num_channels]
            # Warm-up run
            _ = self.model(self.rigraph, x_input)
            # Measure inference time over 10 runs
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                pred = self.model(self.rigraph, x_input)
                # Ensure all CUDA kernels have finished before stopping the timer
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
            print(f"Average inference time over 10 runs (batch size = 1): {avg_time:.6f} seconds")

class StaticTrainer_unstructured(StaticTrainer):
    """
    Trainer for static problems, i.e. problems that do not depend on time and is unstructured.
    """

    def __init__(self, args):
        super().__init__(args)
   
    def init_dataset(self, dataset_config):
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        self.poseidon_dataset_name = ["Poisson-Gauss"]

        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            # Load x
            if self.metadata.group_x is not None and self.metadata.fix_x == False:
                x_array = ds[self.metadata.group_x].values
                if x_array.shape[0] == u_array.shape[0]:
                   x_array = x_array
                   self.x_train = x_array    # [num_samples, num_timesteps, num_nodes, num_dims]
            else:
                raise ValueError("fix_x must be False for unstructured data")
                
        
        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            self.x_train = self.x_train[:,:,:9216,:]

        #c_array = np.concatenate((c_array,self.x_train), axis = -1)
        
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = c_array.shape[-1]
        self.num_output_channels = u_array.shape[-1]

        # Compute dataset sizes
        total_samples = u_array.shape[0]
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."
    
        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = u_array[train_indices]
        u_val = u_array[val_indices]
        u_test = u_array[test_indices]
        x_train = self.x_train[train_indices]
        x_val = self.x_train[val_indices]
        x_test = self.x_train[test_indices]

        if c_array is not None:
            c_train = c_array[train_indices]
            c_val = c_array[val_indices]
            c_test = c_array[test_indices]
        else:
            c_train = c_val = c_test = None
        
        # Compute dataset statistics from training data
        # Reshape u_train to [num_samples * num_timesteps * num_nodes, num_active_vars]
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON  # Avoid division by zero
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)
        # Normalize the data
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std
        # If c is used, compute statistics and normalize c
        if c_array is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + EPSILON  # Avoid division by zero
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)
            # Normalize c
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std



        c_train, u_train, x_train = torch.tensor(c_train, dtype=self.dtype), torch.tensor(u_train, dtype=self.dtype), torch.tensor(x_train, dtype=self.dtype)
        c_val, u_val, x_val = torch.tensor(c_val, dtype=self.dtype), torch.tensor(u_val, dtype=self.dtype), torch.tensor(x_val, dtype=self.dtype)
        c_test, u_test, x_test = torch.tensor(c_test, dtype=self.dtype), torch.tensor(u_test, dtype=self.dtype), torch.tensor(x_test, dtype=self.dtype)
        train_ds = TensorDataset(c_train, u_train, x_train)
        val_ds = TensorDataset(c_val, u_val, x_val)
        test_ds = TensorDataset(c_test, u_test, x_test)

        self.train_loader = DataLoader(train_ds, batch_size=dataset_config.batch_size, shuffle=dataset_config.shuffle, num_workers=dataset_config.num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=dataset_config.batch_size, shuffle=dataset_config.shuffle, num_workers=dataset_config.num_workers)
        self.test_loader = DataLoader(test_ds, batch_size=dataset_config.batch_size, shuffle=dataset_config.shuffle, num_workers=dataset_config.num_workers)

    def init_graph(self, graph_config):
        self.rigraph = RegionInteractionGraph.from_point_cloud(points = torch.tensor(self.x_train[0][0],dtype=self.dtype),
                                              phy_domain=self.metadata.domain_x,
                                              **shallow_asdict(graph_config)
                                            )
        # record the number of edges
        self.config.datarow['p2r edges'] = self.rigraph.physical_to_regional.num_edges
        self.config.datarow['r2r edges'] = self.rigraph.regional_to_regional.num_edges
        self.config.datarow['r2p edges'] = self.rigraph.regional_to_physical.num_edges
                                                    
    def init_model(self, model_config):
        self.model = init_model_from_rigraph(rigraph=self.rigraph, 
                                            input_size=self.num_input_channels, 
                                            output_size=self.num_output_channels, 
                                            model=model_config.name,
                                            drop_edge=model_config.drop_edge,
                                            variable_mesh=model_config.variable_mesh,
                                            config=model_config.args
                                            )

    def train_step(self, batch):
        x_batch, y_batch, coord_batch = batch
        x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device)
        x_batch, y_batch, coord_batch = x_batch.squeeze(1), y_batch.squeeze(1), coord_batch.squeeze(1)
        pred = self.model(self.rigraph, coord_batch, x_batch)
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, coord_batch in loader:
                x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device)
                x_batch, y_batch, coord_batch = x_batch.squeeze(1), y_batch.squeeze(1), coord_batch.squeeze(1)
                pred = self.model(self.rigraph, coord_batch, x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample, coord_sample) in enumerate(self.test_loader):
                x_sample, y_sample, coord_sample = x_sample.to(self.device), y_sample.to(self.device), coord_sample.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                x_sample, y_sample, coord_sample = x_sample.squeeze(1), y_sample.squeeze(1), coord_sample.squeeze(1)
                pred = self.model(self.rigraph, coord_sample, x_sample)
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_sample_de_norm = y_sample * self.u_std.to(self.device) + self.u_mean.to(self.device)
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (direct)"] = final_metric
        print(f"relative error: {final_metric}")

        x_plot = x_sample * self.c_std.to(self.device)  

        fig = plot_estimates(
            u_inp = x_sample[-1].cpu().numpy(), 
            u_gtr = y_sample_de_norm[-1].cpu().numpy(), 
            u_prd = pred_de_norm[-1].cpu().numpy(), 
            x_inp = coord_sample[0].cpu().numpy(),
            x_out = coord_sample[0].cpu().numpy(),
            names = self.metadata.names['u'],
            symmetric = self.metadata.signed['u'],
            domain = self.metadata.domain_x)

        fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


class StaticTrainer_test(StaticTrainer):
    """
    Trainer for static problems, i.e. problems that do not depend on time.
    """

    def __init__(self, args):
        super().__init__(args)
   
    def init_dataset(self, dataset_config):
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        self.poseidon_dataset_name = ["Poisson-Gauss"]

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
                x_array = ds[self.metadata.group_x].values
                if x_array.shape[0] == u_array.shape[0]:
                   x_array = x_array
                self.x_train = x_array
            else:
                domain_x = self.metadata.domain_x #([xmin, ymin], [xmax, ymax])
                nx, ny = u_array.shape[-2], u_array.shape[-1]
                x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
                x_grid = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
                x_grid = x_grid.reshape(-1, 2)  # [num_nodes, num_dims]
                x_grid = x_grid[None, None, ...]  # Add sample and time dimensions
                self.x_train = x_grid # store the x array for later use
                c_array = c_array.reshape(c_array.shape[0],-1)[:,None,:,None] # [num_samples, 1, num_nodes, 1]
                u_array = u_array.reshape(u_array.shape[0],-1)[:,None,:,None] # [num_samples, 1, num_nodes, 1]
                

        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            self.x_train = self.x_train[:,:,:9216,:]
        
        # concatenate
        c_array = np.concatenate((c_array,self.x_train), axis = -1)

        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = c_array.shape[-1]
        self.num_output_channels = u_array.shape[-1]

        # Compute dataset sizes
        total_samples = u_array.shape[0]
        train_size = dataset_config["train_size"]
        val_size = dataset_config["val_size"]
        test_size = dataset_config["test_size"]
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."
    
        # Split data into train, val, test
        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = u_array[train_indices]
        u_val = u_array[val_indices]
        u_test = u_array[test_indices]

        if c_array is not None:
            c_train = c_array[train_indices]
            c_val = c_array[val_indices]
            c_test = c_array[test_indices]
        else:
            c_train = c_val = c_test = None
        
        # Compute dataset statistics from training data
        # Reshape u_train to [num_samples * num_timesteps * num_nodes, num_active_vars]
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON  # Avoid division by zero

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
            c_std = np.std(c_train_flat, axis=0) + EPSILON  # Avoid division by zero

            # Store statistics
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)

            # Normalize c
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std

        self.x_train = torch.tensor(self.x_train, dtype=self.dtype).to(self.device)
        
        c_train, u_train = torch.tensor(c_train, dtype=self.dtype), torch.tensor(u_train, dtype=self.dtype)
        c_val, u_val = torch.tensor(c_val, dtype=self.dtype), torch.tensor(u_val, dtype=self.dtype)
        c_test, u_test = torch.tensor(c_test, dtype=self.dtype), torch.tensor(u_test, dtype=self.dtype)
        train_ds = TensorDataset(c_train, u_train)
        val_ds = TensorDataset(c_val, u_val)
        test_ds = TensorDataset(c_test, u_test)

        self.train_loader = DataLoader(train_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.val_loader = DataLoader(val_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.test_loader = DataLoader(test_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])

    def init_graph(self, graph_config):
        self.rigraph = RegionInteractionGraph.from_point_cloud(points = self.x_train[0][0],
                                              phy_domain=self.metadata.domain_x,
                                              **shallow_asdict(graph_config)
                                            )
        # record the number of edges
        self.config.datarow['p2r edges'] = self.rigraph.physical_to_regional.num_edges
        self.config.datarow['r2r edges'] = self.rigraph.regional_to_regional.num_edges
        self.config.datarow['r2p edges'] = self.rigraph.regional_to_physical.num_edges
                                                    
    def init_model(self, model_config):
        self.model = init_model_from_rigraph(rigraph=self.rigraph, 
                                            input_size=self.num_input_channels, 
                                            output_size=self.num_output_channels, 
                                            model=model_config.name,
                                            drop_edge=model_config.drop_edge,
                                            variable_mesh=model_config.variable_mesh,
                                            config=model_config.args
                                            )
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample) in enumerate(self.test_loader):
                x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                x_sample, y_sample = x_sample.squeeze(1), y_sample.squeeze(1)
                pred = self.model(self.rigraph, x_sample)
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_sample_de_norm = y_sample * self.u_std.to(self.device) + self.u_mean.to(self.device)
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (direct)"] = final_metric

        x_plot = x_sample * self.c_std.to(self.device)  
        self.plot_results(self.rigraph.physical_to_regional.src_ndata['pos'], x_sample[0], y_sample_de_norm[0], pred_de_norm[0])
        
    def plot_results(self, coords, input, gt, pred):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        im1 = ax1.tricontourf(coords[:, 0].cpu(), coords[:, 1].cpu(), input[:, 0].cpu(), cmap='plasma')
        ax1.set_title("Input (c)")
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.tricontourf(coords[:, 0].cpu(), coords[:, 1].cpu(), gt[:, 0].cpu(), cmap='plasma')
        ax2.set_title("Ground Truth (u)")
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)

        im3 = ax3.tricontourf(coords[:, 0].cpu(), coords[:, 1].cpu(), pred[:, 0].cpu(), cmap='plasma')
        ax3.set_title("Prediction")
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.savefig(self.path_config["result_path"])
        plt.close()
        
    def measure_inference_time(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            # Get a single sample from the test dataset
            x_sample, y_sample = self.test_loader.dataset[0]
            # Ensure inputs are tensors and add batch dimension
            x_sample = x_sample.to(self.device).unsqueeze(0)  # Shape: [1, num_timesteps, num_nodes, num_channels]
            # Since it's a static problem, squeeze the time dimension
            x_input = x_sample.squeeze(1)  # Shape: [1, num_nodes, num_channels]
            # Warm-up run
            _ = self.model(self.rigraph, x_input)
            # Measure inference time over 10 runs
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                pred = self.model(self.rigraph, x_input)
                # Ensure all CUDA kernels have finished before stopping the timer
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
            print(f"Average inference time over 10 runs (batch size = 1): {avg_time:.6f} seconds")
