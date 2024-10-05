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

from src.data.dataset import Metadata, DATASET_METADATA
from src.graph import RegionInteractionGraph
from src.model import init_model_from_rigraph



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
                   x_array = x_array[0:1]
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

        if dataset_name in self.poseidon_dataset_name:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            self.x_train = self.x_train[:,:,:9216,:]
        
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = c_array.shape[-1]
        self.num_output_channels = u_array.shape[-1]
        
        #self.plot_results(torch.self.x_train[0][0], torch.tensor(c_array[0][0]), torch.tensor(u_array[0][0]), torch.tensor(u_array[0][0]))
        #assert self.model_config["args"]["in_channels"] == num_input_channels, f"Expected {num_input_channels} input channels, but found {self.model_config['args']['in_channels']}."
 
        total_samples = u_array.shape[0]
        train_size, val_size, test_size = dataset_config["train_size"], dataset_config["val_size"], dataset_config["test_size"]
        
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."
        
        self.x_train = torch.tensor(self.x_train, dtype=self.dtype).to(self.device)
        train_ds = TensorDataset(torch.tensor(c_array[:train_size], dtype=self.dtype), torch.tensor(u_array[:train_size], dtype=self.dtype))
        val_ds = TensorDataset(torch.tensor(c_array[train_size:train_size+val_size], dtype=self.dtype), torch.tensor(u_array[train_size:train_size+val_size], dtype=self.dtype))
        test_ds = TensorDataset(torch.tensor(c_array[-test_size:], dtype=self.dtype), torch.tensor(u_array[-test_size:], dtype=self.dtype))

        self.train_loader = DataLoader(train_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.val_loader = DataLoader(val_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.test_loader = DataLoader(test_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])

    def init_graph(self, graph_config):
        graph_config = OmegaConf.to_container(graph_config, resolve=True)
        self.rigraph = RegionInteractionGraph.from_point_cloud(points = self.x_train[0][0],
                                              phy_domain=self.metadata.domain_x,
                                              **graph_config
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
                relative_errors = compute_batch_errors(y_sample, pred, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (poseidon_metric)"] = final_metric

        self.plot_results(self.rigraph.physical_to_regional.src_ndata['pos'], x_sample[0], y_sample[0], pred[0])
            
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
            x_sample, y_sample, coord_sample = self.test_loader.dataset[0]
            # Ensure inputs are tensors and add batch dimension
            x_sample = x_sample.to(self.device).unsqueeze(0)  # Shape: [1, num_timesteps, num_nodes, num_channels]
            coord_sample = coord_sample.to(self.device).unsqueeze(0)  # Shape: [1, num_timesteps, num_nodes, num_dims]
            # Since it's a static problem, squeeze the time dimension
            x_input = x_sample.squeeze(1)  # Shape: [1, num_nodes, num_channels]
            coord_input = coord_sample.squeeze(1)  # Shape: [1, num_nodes, num_dims]
            # Prepare input_geom and output_queries
            input_geom = coord_input[0:1]  # Shape: [1, num_nodes, num_dims]
            output_queries = coord_input[0]  # Shape: [num_nodes, num_dims]
            # Warm-up run
            _ = self.model(self.rigraph, x_sample)
            # Measure inference time over 10 runs
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                pred = self.model(self.rigraph, x_sample)
                # Ensure all CUDA kernels have finished before stopping the timer
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
            print(f"Average inference time over 10 runs (batch size = 1): {avg_time:.6f} seconds")