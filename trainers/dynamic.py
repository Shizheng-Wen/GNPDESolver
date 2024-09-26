import os 
import torch
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as np
from tqdm import tqdm

from .base import TrainerBase
from .utils import manual_seed, create_all_to_all_pairs, DynamicPairDataset_half, TestDataset, compute_batch_errors, compute_final_metric
from architectures.gino import GINO
from data.dataset import Metadata, DATASET_METADATA

#from architectures.rigno.model import RIGNO
#from architectures.rigno.graph import RegionInteractionGraph, add_dummy, remove_dummy



import matplotlib.pyplot as plt
import matplotlib.tri as tri

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
        self.poseidon_dataset_name = ["CE-RP","CE-Gauss",
                                      "NS-PwC","NS-SVS","NS-Gauss","NS-SL",
                                       "ACE", "Wave-Layer"]

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
                self.x_train = x_array # store the x array for later use
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
                self.x_train = x_grid # store the x array for later use
        
        if dataset_name in self.poseidon_dataset_name:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            self.x_train = self.x_train[:,:,:9216,:]
        
        self.x_train = torch.tensor(self.x_train, dtype=self.dtype).to(self.device)
        # Handle active variables
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]  # slice is more efficient than indexing
    
        # Compute dataset sizes
        total_samples = u_array.shape[0]
        train_size = dataset_config["train_size"]
        val_size = dataset_config["val_size"]
        test_size = dataset_config["test_size"]

        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"

        # Split data into train, val, test
        u_train = u_array[:train_size]
        u_val = u_array[train_size:train_size+val_size]
        u_test = u_array[-test_size:]

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

        if self.metadata.domain_t is not None:
            t_start, t_end = self.metadata.domain_t
            t_values = np.linspace(t_start, t_end, u_array.shape[1]) # shape: [num_timesteps]
        else:
            raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")
    
        max_time_diff = dataset_config.get("max_time_diff", None)
        
        self.train_dataset = DynamicPairDataset_half(u_train, c_train, t_values, self.metadata, max_time_diff = max_time_diff)
        
        self.time_stats = self.train_dataset.time_stats
        
        self.val_dataset = DynamicPairDataset_half(u_val, c_val, t_values, self.metadata, max_time_diff = max_time_diff, time_stats=self.time_stats)
        self.test_dataset = DynamicPairDataset_half(u_test, c_test, t_values, self.metadata, max_time_diff = max_time_diff, time_stats=self.time_stats)

        batch_size = dataset_config["batch_size"]
        shuffle = dataset_config["shuffle"]
        num_workers = dataset_config["num_workers"]
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn)

        # Generate latent queries (if needed)
        x_min, y_min = self.metadata.domain_x[0]
        x_max, y_max = self.metadata.domain_x[1]
        meshgrid = np.meshgrid(np.linspace(x_min, x_max, dataset_config["latent_queries"][0]),
                            np.linspace(y_min, y_max, dataset_config["latent_queries"][1]),
                            indexing='ij')
        latent_queries = np.stack(meshgrid, axis=-1)  # [\sqrt{num_queries}, \sqrt{num_queries}, 2]
        self.latent_queries = torch.tensor(latent_queries, dtype=self.dtype).unsqueeze(0).to(self.device)  # [1, num_queries, 2]

    def collate_fn(self, batch):
        input_list, output_list = zip(*batch) # unzip the batch, both inputs and outputs are lists of tuples
        inputs = np.stack(input_list) # shape: [batch_size, num_nodes, input_dim]
        outputs = np.stack(output_list) # shape: [batch_size, num_nodes, output_dim]

        inputs = torch.tensor(inputs, dtype=self.dtype)
        outputs = torch.tensor(outputs, dtype=self.dtype)

        return inputs, outputs

    def init_model(self, model_config):
        in_channels = self.u_mean.shape[0] + 2

        if hasattr(self, 'c_mean'):
            in_channels += self.c_mean.shape[0]

        out_channels = self.u_mean.shape[0]

        self.model = GINO(in_channels=in_channels, out_channels=out_channels, **model_config["args"])

    def train_step(self, batch):
        batch_inputs, batch_outputs = batch
        batch_inputs, batch_outputs = batch_inputs.to(self.device), batch_outputs.to(self.device) # Shape: [batch_size, num_nodes, num_channels]
        pred = self.model(x=batch_inputs, input_geom=self.x_train[0], latent_queries=self.latent_queries, output_queries=self.x_train[0][0])
        return self.loss_fn(pred, batch_outputs)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(x=x_batch, input_geom=self.x_train[0], latent_queries=self.latent_queries, output_queries=self.x_train[0][0])
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)
    
    def autoregressive_predict(self, x_batch, time_indices):
        """
        Autoregressive prediction of the output variables at the specified time indices.

        Args:
            x_batch (torch.Tensor): Initial input batch at time t=0. Shape: [batch_size, num_nodes, input_dim]
            time_indices (np.ndarray): Array of time indices for prediction (e.g., [0, 2, 4, ..., 14])

        Returns:
            torch.Tensor: Predicted outputs over time. Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        """
        
        batch_size, num_nodes, input_dim = x_batch.shape
        num_timesteps = len(time_indices)

        predictions = []

        t_values = self.test_dataset.t_values
        start_times_mean = self.time_stats['start_times_mean']
        start_times_std = self.time_stats['start_times_std']
        time_diffs_mean = self.time_stats['time_diffs_mean']
        time_diffs_std = self.time_stats['time_diffs_std']

        u_in_dim = self.u_mean.shape[0]
        c_in_dim = self.c_mean.shape[0] if hasattr(self, 'c_mean') else 0
        time_feature_dim = 2

        if c_in_dim > 0:
            c_in = x_batch[..., u_in_dim:u_in_dim+c_in_dim] # Shape: [batch_size, num_nodes, c_in_dim]
        else:
            c_in = None
        
        current_u_in = x_batch[..., :u_in_dim] # Shape: [batch_size, num_nodes, u_in_dim]
        
        for idx in range(1, num_timesteps):
            t_in_idx = time_indices[idx-1]
            t_out_idx = time_indices[idx]
            start_time = t_values[t_in_idx]
            time_diff = t_values[t_out_idx] - t_values[t_in_idx]
            
            start_time_norm = (start_time - start_times_mean) / start_times_std
            time_diff_norm = (time_diff - time_diffs_mean) / time_diffs_std

            # Prepare time features (expanded to match num_nodes)
            start_time_expanded = torch.full((batch_size, num_nodes, 1), start_time_norm, dtype=self.dtype).to(self.device)
            time_diff_expanded = torch.full((batch_size, num_nodes, 1), time_diff_norm, dtype=self.dtype).to(self.device)

            input_features = [current_u_in]  # Use the previous u_in (either initial or previous prediction)
            if c_in is not None:
                input_features.append(c_in)  # Use the same c_in as in x_batch (assumed constant over time)
            input_features.append(start_time_expanded)
            input_features.append(time_diff_expanded)
            x_input = torch.cat(input_features, dim=-1)  # Shape: [batch_size, num_nodes, input_dim]

            # Forward pass
            with torch.no_grad():
                pred = self.model(x=x_input, input_geom=self.x_train[0],
                                latent_queries=self.latent_queries, output_queries=self.x_train[0][0])
                # pred shape: [batch_size, num_nodes, output_dim]

            # Store prediction
            predictions.append(pred)

            # Update current_u_in for next iteration
            current_u_in = pred
        
        predictions = torch.stack(predictions, dim=1) # Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        
        return predictions
        
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        if self.dataset_config["predict_mode"] == "autoregressive":
            time_indices = np.arange(0, 15, 2) # [0, 2, 4, ..., 14]
        elif self.dataset_config["predict_mode"] == "direct":
            time_indices = np.array([0,14])
        
        test_dataset = TestDataset(
            u_data = self.test_dataset.u_data,
            c_data = self.test_dataset.c_data,
            t_values = self.test_dataset.t_values,
            metadata = self.metadata,
            time_indices = time_indices
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.test_loader.batch_size,
            shuffle=False,
            num_workers=self.test_loader.num_workers,
            collate_fn=self.collate_fn
        )

        pbar = tqdm(total=len(test_loader), desc="Testing", colour="blue")
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_loader):
                # TODO: Figure out whether from CPU to GPU is the compuation bottleneck
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device) # Shape: [batch_size, num_nodes, num_channels]
                pred = self.autoregressive_predict(x_batch, time_indices) # Shape: [batch_size, num_timesteps - 1, num_nodes, num_channels]
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_batch_de_norm = y_batch * self.u_std.to(self.device) + self.u_mean.to(self.device)
                if self.dataset_config["metric"] == "final_step":
                    relative_errors = compute_batch_errors(y_batch_de_norm[:,-2:-1,:,:], pred_de_norm[:,-2:-1,:,:], self.metadata)
                elif self.dataset_config["metric"] == "all_step":
                    relative_errors = compute_batch_errors(y_batch_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
                pbar.update(1)
        pbar.close()
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (poseidon_metric)"] = final_metric
        self.plot_results(
            coords=self.x_train[0,0].cpu().numpy(),
            gt_sequence=y_batch_de_norm[0].cpu().numpy(),
            pred_sequence=pred_de_norm[0].cpu().numpy(),
            time_indices=time_indices,
            t_values=self.test_dataset.t_values,
            num_frames=5
        )

    def plot_results(self, coords, gt_sequence, pred_sequence, time_indices, t_values, num_frames=5):
        """
        Plots several frames of ground truth and predicted results using contour plots for all variables.

        Args:
            coords (numpy.ndarray): Coordinates of the nodes. Shape: [num_nodes, num_dims]
            gt_sequence (numpy.ndarray): Ground truth sequence. Shape: [num_timesteps, num_nodes, num_vars]
            pred_sequence (numpy.ndarray): Predicted sequence. Shape: [num_timesteps, num_nodes, num_vars]
            time_indices (np.ndarray): Array of time indices used in the prediction.
            t_values (np.ndarray): Actual time values corresponding to the time indices.
            num_frames (int): Number of frames to plot. Defaults to 5.
        """
        num_timesteps = gt_sequence.shape[0]
        num_nodes = coords.shape[0]
        num_vars = gt_sequence.shape[-1]

        # Select frames to plot
        frame_indices = np.linspace(0, num_timesteps - 1, num_frames, dtype=int)

        x = coords[:, 0]
        y = coords[:, 1]

        # Create a figure with num_vars * 3 rows and num_frames columns
        fig_height = 3 * num_vars * 4
        fig, axes = plt.subplots(num_vars * 3, num_frames, figsize=(4 * num_frames, fig_height))

        # Ensure axes is a 2D array
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape((num_vars * 3, num_frames))

        # Compute vmin and vmax per variable
        vmin_list = []
        vmax_list = []
        for variable_idx in range(num_vars):
            min_val = min(gt_sequence[:, :, variable_idx].min(), pred_sequence[:, :, variable_idx].min())
            max_val = max(gt_sequence[:, :, variable_idx].max(), pred_sequence[:, :, variable_idx].max())
            vmin_list.append(min_val)
            vmax_list.append(max_val)

        for variable_idx in range(num_vars):
            vmin = vmin_list[variable_idx]
            vmax = vmax_list[variable_idx]
            for i, frame_idx in enumerate(frame_indices):
                time_idx = time_indices[frame_idx + 1]  # +1 because gt_sequence and pred_sequence start from time_indices[1:]
                time_value = t_values[time_idx]

                gt = gt_sequence[frame_idx][:, variable_idx]
                pred = pred_sequence[frame_idx][:, variable_idx]
                abs_error = np.abs(gt - pred)

                # Row indices
                row_gt = variable_idx * 3
                row_pred = variable_idx * 3 + 1
                row_error = variable_idx * 3 + 2

                # Ground Truth
                ax_gt = axes[row_gt, i]
                ct_gt = ax_gt.tricontourf(x, y, gt, cmap='RdBu', vmin=vmin, vmax=vmax)
                if i == 0:
                    ax_gt.set_ylabel(f'Variable {variable_idx + 1} - Ground Truth')
                if variable_idx == 0:
                    ax_gt.set_title(f"Time: {time_value:.2f}")
                ax_gt.set_aspect('equal')
                plt.colorbar(ct_gt, ax=ax_gt)

                # Prediction
                ax_pred = axes[row_pred, i]
                ct_pred = ax_pred.tricontourf(x, y, pred, cmap='RdBu', vmin=vmin, vmax=vmax)
                if i == 0:
                    ax_pred.set_ylabel('Prediction')
                ax_pred.set_aspect('equal')
                plt.colorbar(ct_pred, ax=ax_pred)

                # Absolute Error
                ax_error = axes[row_error, i]
                ct_error = ax_error.tricontourf(x, y, abs_error, cmap='hot')
                if i == 0:
                    ax_error.set_ylabel('Absolute Error')
                ax_error.set_aspect('equal')
                plt.colorbar(ct_error, ax=ax_error)

        plt.tight_layout()
        plt.savefig(self.path_config["result_path"])
        plt.close()

    def measure_inference_time(self):
        import time
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            # Get a single sample from the test dataset
            x_sample, y_sample = self.test_dataset[0]
            # Ensure inputs are tensors and add batch dimension
            x_sample = torch.tensor(x_sample, dtype=self.dtype).unsqueeze(0).to(self.device)  # Shape: [1, num_nodes, input_dim]
            # Warm-up run
            _ = self.model(
                x=x_sample,
                input_geom=self.x_train[0],
                latent_queries=self.latent_queries,
                output_queries=self.x_train[0][0]
            )
            # Measure inference time over 10 runs
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                pred = self.model(
                    x=x_sample,
                    input_geom=self.x_train[0],
                    latent_queries=self.latent_queries,
                    output_queries=self.x_train[0][0]
                )
                # Ensure all CUDA kernels have finished before stopping the timer
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
            print(f"Average inference time over 10 runs (batch size = 1): {avg_time:.6f} seconds")

class RIGNOTrainer(DynamicTrainer):
    def init_model(self, model_config):
        in_channels = self.u_mean.shape[0] + 2

        if hasattr(self, 'c_mean'):
            in_channels += self.c_mean.shape[0]

        out_channels = self.u_mean.shape[0]
        self.rigraph = RegionInteractionGraph.from_point_cloud(self.x_train[0][0],
                                                          output_points=None,
                                                          periodic=False,
                                                          sample_factor=model_config["args"]["sample_factor"],
                                                          overlap_factor_p2r=model_config["args"]["overlap_factor_p2r"],
                                                          overlap_factor_r2p=model_config["args"]["overlap_factor_r2p"],
                                                          regional_level=model_config["args"]["regional_level"],
                                                          )
        self.model = RIGNO(
            node_input_size=self.rigraph.physical_to_regional.ndim[0] + in_channels, # +1 for time
            edge_encoder_input_size=self.rigraph.physical_to_regional.edim,
            edge_processor_input_size=self.rigraph.regional_to_regional.edim,
            edge_decoder_input_size=self.rigraph.regional_to_physical.edim,
            output_size=out_channels,
        )

    def train_step(self, batch):
        batch_inputs, batch_outputs = batch
        batch_inputs, batch_outputs = batch_inputs.to(self.device), batch_outputs.to(self.device) # Shape: [batch_size, num_nodes, num_channels]
        pred = self.model(self.rigraph, add_dummy(batch_inputs[0]))
        return self.loss_fn(pred, batch_outputs)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(self.rigraph, x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def autoregressive_predict(self, x_batch, time_indices):
        """
        Autoregressive prediction of the output variables at the specified time indices.

        Args:
            x_batch (torch.Tensor): Initial input batch at time t=0. Shape: [batch_size, num_nodes, input_dim]
            time_indices (np.ndarray): Array of time indices for prediction (e.g., [0, 2, 4, ..., 14])

        Returns:
            torch.Tensor: Predicted outputs over time. Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        """
        
        batch_size, num_nodes, input_dim = x_batch.shape
        num_timesteps = len(time_indices)

        predictions = []

        t_values = self.test_dataset.t_values
        start_times_mean = self.time_stats['start_times_mean']
        start_times_std = self.time_stats['start_times_std']
        time_diffs_mean = self.time_stats['time_diffs_mean']
        time_diffs_std = self.time_stats['time_diffs_std']

        u_in_dim = self.u_mean.shape[0]
        c_in_dim = self.c_mean.shape[0] if hasattr(self, 'c_mean') else 0
        time_feature_dim = 2

        if c_in_dim > 0:
            c_in = x_batch[..., u_in_dim:u_in_dim+c_in_dim] # Shape: [batch_size, num_nodes, c_in_dim]
        else:
            c_in = None
        
        current_u_in = x_batch[..., :u_in_dim] # Shape: [batch_size, num_nodes, u_in_dim]
        
        for idx in range(1, num_timesteps):
            t_in_idx = time_indices[idx-1]
            t_out_idx = time_indices[idx]
            start_time = t_values[t_in_idx]
            time_diff = t_values[t_out_idx] - t_values[t_in_idx]
            
            start_time_norm = (start_time - start_times_mean) / start_times_std
            time_diff_norm = (time_diff - time_diffs_mean) / time_diffs_std

            # Prepare time features (expanded to match num_nodes)
            start_time_expanded = torch.full((batch_size, num_nodes, 1), start_time_norm, dtype=self.dtype).to(self.device)
            time_diff_expanded = torch.full((batch_size, num_nodes, 1), time_diff_norm, dtype=self.dtype).to(self.device)

            input_features = [current_u_in]  # Use the previous u_in (either initial or previous prediction)
            if c_in is not None:
                input_features.append(c_in)  # Use the same c_in as in x_batch (assumed constant over time)
            input_features.append(start_time_expanded)
            input_features.append(time_diff_expanded)
            x_input = torch.cat(input_features, dim=-1)  # Shape: [batch_size, num_nodes, input_dim]

            # Forward pass
            with torch.no_grad():
                pred = self.model(self.rigraph, x_input)
                # pred shape: [batch_size, num_nodes, output_dim]

            # Store prediction
            predictions.append(pred)

            # Update current_u_in for next iteration
            current_u_in = pred
        
        predictions = torch.stack(predictions, dim=1) # Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        
        return predictions


        