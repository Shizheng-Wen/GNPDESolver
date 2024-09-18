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

from .base import TrainerBase
from .utils import manual_seed, compute_batch_errors, compute_final_metric
from architectures.gino import GINO
from data.dataset import Metadata, DATASET_METADATA



class StaticTrainer(TrainerBase):
    """
    Trainer for static problems, i.e. problems that do not depend on time.
    """

    def __init__(self, args):
        self.config = args
        self.setup_config = self.config.setup
        self.dataset_config = self.config.dataset
        self.model_config = self.config.model
        self.optimizer_config = self.config.optimizer
        self.path_config = self.config.path
        self.metadata = DATASET_METADATA[self.dataset_config["metaname"]]

        self.device = self.setup_config["device"]
        manual_seed(self.setup_config["seed"])
        self.dtype = self.setup_config["dtype"]
        self.loss_fn = nn.MSELoss()

        self.init_dataset(self.dataset_config)
        self.init_model(self.model_config)
        self.init_optimizer(self.optimizer_config)

        nparam = sum(
            [p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()]
        )
        nbytes = sum(
            [p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size() for p in self.model.parameters()]
        )
        args.datarow['nparams'] = nparam
        args.datarow['nbytes'] = nbytes

    
    def init_dataset(self, dataset_config):
        base_path = dataset_config["base_path"]
        dataset_path = os.path.join(base_path, f"{dataset_config['name']}.nc")
        ds = xr.open_dataset(dataset_path)

        # Load dataset -> Shape [num_samples, num_timesteps, num_nodes, num_channels]
        u = ds[self.metadata.group_u].values
        u_tensor = torch.tensor(u, dtype=torch.float32)

        if self.metadata.group_c is not None:
            c = ds[self.metadata.group_c].values
            c_tensor = torch.tensor(c, dtype=torch.float32)
        else:
            c_tensor = None

        if self.metadata.group_x is not None:
            x = ds[self.metadata.group_x].values
            x_tensor = torch.tensor(x, dtype=torch.float32) # Shape [num_samples, num_timesteps, num_nodes, num_dims]
        else:
            # generate x coordinates if not available (e.g., for airfoil_grid) -> Shape [num_samples, num_nodes_x, num_nodes_y]
            domain_x = self.metadata.domain_x #([xmin, ymin], [xmax, ymax])
            nx, ny = u_tensor.shape[-2], u_tensor.shape[-1]
            x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
            y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
            xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
            x_grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)[None, ...]  # Shape: [1, nx*ny, 2]
            x_grid = np.broadcast_to(x_grid, (u_tensor.shape[0], nx*ny, 2))  # Shape: [num_samples, num_nodes, 2]
            x_tensor = torch.tensor(x_grid, dtype=torch.float32)
            u_tensor = u_tensor.reshape(u_tensor.shape[0], -1) # [num_samples, num_nodes]
            u_tensor = u_tensor[:, None, :, None] # [num_samples, 1, num_nodes, 1]
            if c_tensor is not None:
                c_tensor = c_tensor.reshape(c_tensor.shape[0], -1) # [num_samples, num_nodes]
                c_tensor = c_tensor[:, None, :, None] # [num_samples, 1, num_nodes, 1]
        brekpoint()
        active_vars = self.metadata.active_variables
        u_tensor = u_tensor[..., active_vars]
        self.num_input_channels = c_tensor.shape[-1]
        self.num_output_channels = u_tensor.shape[-1]
        #assert self.model_config["args"]["in_channels"] == num_input_channels, f"Expected {num_input_channels} input channels, but found {self.model_config['args']['in_channels']}."

        total_samples = u_tensor.shape[0]
        train_size, val_size, test_size = dataset_config["train_size"], dataset_config["val_size"], dataset_config["test_size"]
        
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_tensor.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."

        train_ds = TensorDataset(c_tensor[:train_size], u_tensor[:train_size], x_tensor[:train_size])
        val_ds = TensorDataset(c_tensor[train_size:train_size+val_size], u_tensor[train_size:train_size+val_size], x_tensor[train_size:train_size+val_size])
        test_ds = TensorDataset(c_tensor[-test_size:], u_tensor[-test_size:], x_tensor[-test_size:])

        self.train_loader = DataLoader(train_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.val_loader = DataLoader(val_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])
        self.test_loader = DataLoader(test_ds, batch_size=dataset_config["batch_size"], shuffle=dataset_config["shuffle"], num_workers=dataset_config["num_workers"])

        # Generate latent queries
        x_min, y_min = self.metadata.domain_x[0]
        x_max, y_max = self.metadata.domain_x[1]
        meshgrid = torch.meshgrid(torch.linspace(x_min, x_max, dataset_config["latent_queries"][0]), torch.linspace(y_min, y_max, dataset_config["latent_queries"][1]), indexing='ij')
        self.latent_queries = torch.stack(meshgrid, dim=-1).unsqueeze(0).to(self.device) # [1, 64, 64, 2]

    def init_model(self, model_config):
        self.model = GINO(in_channels=self.num_input_channels, out_channels=self.num_output_channels, **model_config["args"])
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample, coord_sample) in enumerate(self.test_loader):
                x_sample, y_sample, coord_sample = x_sample.to(self.device), y_sample.to(self.device), coord_sample.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                pred = self.model(x=x_sample.squeeze(1), input_geom=coord_sample.squeeze(1)[0:1], latent_queries=self.latent_queries, output_queries=coord_sample.squeeze(1)[0]).unsqueeze(1) # Shape: [batch_size, 1, num_nodes, num_channels]
                relative_errors = compute_batch_errors(y_sample, pred, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (poseidon_metric)"] = final_metric
        self.plot_results(coord_sample.squeeze(1)[45], x_sample.squeeze(1)[45], y_sample.squeeze(1)[45], pred.squeeze(1)[45])
            
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
        
        


        

        


