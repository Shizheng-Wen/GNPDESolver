import os 
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import xarray as xr
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from .base import TrainerBase
from .utils import (manual_seed, DynamicPairDataset, TestDataset, 
                    compute_batch_errors, compute_final_metric)
from .utils.io_norm import compute_stats
from .utils.data_pairs import CombinedDataLoader
from .utils.plot import plot_estimates, plot_error_vs_time, visualize_encoder_output
from ..utils import shallow_asdict


from src.data.dataset import Metadata, DATASET_METADATA
from src.graph import RegionInteractionGraph
from src.model import init_model_from_rigraph


EPSILON = 1e-10

class SequentialTrainer(TrainerBase):
    """
    Trainer for sequential data using a single dataset and a neural operator model.
    """

    def __init__(self, args):
        """
        Initialize the SequentialTrainer.

        Args:
            args (Namespace): Configuration arguments for the trainer.
        """
        super().__init__(args)
    
    def init_dataset(self, dataset_config):
        """
        Initialize the dataset for training, validation, and testing.

        Args:
            dataset_config (DatasetConfig): Configuration for the dataset, including paths, sizes, and other settings.

        Returns:
            None
        """

        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        self.poseidon_dataset_name = ["CE-RP","CE-Gauss","NS-PwC",
                                      "NS-SVS","NS-Gauss","NS-SL",
                                      "ACE", "Wave-Layer"]
        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            if self.metadata.group_x is not None:
                x_array = ds[self.metadata.group_x].values  # Shape: [1, 1, num_nodes, num_dims]
                self.x_train = x_array 
            else:
                # Generate x coordinates if not available, but sequential data doesn't need it currenty  (e.g., for structured grids)
                domain_x = self.metadata.domain_x  # ([xmin, ymin], [xmax, ymax])
                nx, ny = u_array.shape[2], u_array.shape[3]  # Spatial dimensions
                x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')  # [nx, ny]
                x_grid = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
                x_grid = x_grid.reshape(-1, 2)  # [num_nodes, num_dims]
                x_grid = x_grid[None, None, ...]  # Add sample and time dimensions
                self.x_train = x_grid # store the x array for later use
        
        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
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
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
    
        # Split data into train, val, test
        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = np.ascontiguousarray(u_array[train_indices])
        u_val = np.ascontiguousarray(u_array[val_indices])
        u_test = np.ascontiguousarray(u_array[test_indices])

        if c_array is not None:
            c_train = np.ascontiguousarray(c_array[train_indices])
            c_val = np.ascontiguousarray(c_array[val_indices])
            c_test = np.ascontiguousarray(c_array[test_indices])
        else:
            c_train = c_val = c_test = None

        if self.metadata.domain_t is not None:
            t_start, t_end = self.metadata.domain_t
            t_values = np.linspace(t_start, t_end, u_array.shape[1]) # shape: [num_timesteps]
        else:
            raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")

        max_time_diff = getattr(dataset_config, "max_time_diff", None)
        self.stats = compute_stats(u_train, c_train, t_values, self.metadata,max_time_diff,
                                  sample_rate=dataset_config.sample_rate,
                                  use_metadata_stats=dataset_config.use_metadata_stats,
                                  use_time_norm=dataset_config.use_time_norm)
        self.train_dataset = DynamicPairDataset(u_train, c_train, t_values, self.metadata, 
                                                max_time_diff = max_time_diff, 
                                                stepper_mode=dataset_config.stepper_mode,
                                                stats=self.stats,
                                                use_time_norm = dataset_config.use_time_norm)
        self.val_dataset = DynamicPairDataset(u_val, c_val, t_values, self.metadata, 
                                              max_time_diff = max_time_diff, 
                                              stepper_mode=dataset_config.stepper_mode,
                                              stats=self.stats,
                                              use_time_norm = dataset_config.use_time_norm)
        self.test_dataset = DynamicPairDataset(u_test, c_test, t_values, self.metadata, 
                                               max_time_diff = max_time_diff, 
                                               stepper_mode=dataset_config.stepper_mode,
                                               stats=self.stats,
                                               use_time_norm = dataset_config.use_time_norm)
        batch_size = dataset_config.batch_size
        shuffle = dataset_config.shuffle
        num_workers = dataset_config.num_workers
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        Custom collate function to prepare batches for the DataLoader.

        Args:
            batch (list): List of tuples where each tuple contains input and output arrays.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing batched inputs and outputs as tensors.
        """

        input_list, output_list = zip(*batch) # unzip the batch, both inputs and outputs are lists of tuples
        inputs = np.stack(input_list) # shape: [batch_size, num_nodes, input_dim]
        outputs = np.stack(output_list) # shape: [batch_size, num_nodes, output_dim]

        inputs = torch.tensor(inputs, dtype=self.dtype)
        outputs = torch.tensor(outputs, dtype=self.dtype)

        return inputs, outputs

    def init_graph(self, graph_config):
        """
        Initialize the region interaction graph (RIGraph) for the model.

        Args:
            graph_config (GraphConfig): Configuration for the graph, including parameters like radius and edge attributes.

        Returns:
            None
        """

        self.rigraph = RegionInteractionGraph.from_point_cloud(points = self.x_train[0][0],
                                              phy_domain=self.metadata.domain_x,
                                              **shallow_asdict(graph_config)
                                            )
        # record the number of edges
        self.config.datarow['p2r edges'] = self.rigraph.physical_to_regional.num_edges
        self.config.datarow['r2r edges'] = self.rigraph.regional_to_regional.num_edges
        self.config.datarow['r2p edges'] = self.rigraph.regional_to_physical.num_edges

    def init_model(self, model_config):
        """
        Initialize the neural operator model (Encoder-Processor-Decoder).

        Args:
            model_config (ModelConfig): Configuration for the model, including model name, edge dropout rate, and other settings.

        Returns:
            None
        """

        in_channels = self.stats["u"]["mean"].shape[0] + 2 # add lead time and time difference
        
        if model_config.use_conditional_norm:
            in_channels = in_channels - 1 

        if "c" in self.stats:
            in_channels += self.stats["c"]["mean"].shape[0]

        out_channels = self.stats["u"]["mean"].shape[0]

        self.model = init_model_from_rigraph(rigraph=self.rigraph, 
                                            input_size=in_channels, 
                                            output_size=out_channels, 
                                            model=model_config.name,
                                            drop_edge=model_config.drop_edge,
                                            variable_mesh=model_config.variable_mesh,
                                            config=model_config.args
                                            )

    def train_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing batched inputs and outputs.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """

        self.model.drop_edge = self.model_config.drop_edge
        batch_inputs, batch_outputs = batch
        batch_inputs, batch_outputs = batch_inputs.to(self.device), batch_outputs.to(self.device) # Shape: [batch_size, num_nodes, num_channels]
        if self.model_config.use_conditional_norm:
            pred = self.model(self.rigraph, batch_inputs[...,:-1], batch_inputs[...,0,-2:-1]) # [batch_size, num_nodes, num_channels]
        else:
            pred = self.model(self.rigraph, batch_inputs)

        return self.loss_fn(pred, batch_outputs)
    
    def validate(self, loader):
        """
        Validate the model on a given dataset loader.

        Args:
            loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            float: The average loss over the validation dataset.
        """

        self.model.eval()
        self.model.drop_edge = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_outputs in loader:
                batch_inputs, batch_outputs = batch_inputs.to(self.device), batch_outputs.to(self.device)
                if self.model_config.use_conditional_norm:
                    pred = self.model(self.rigraph, batch_inputs[...,:-1], batch_inputs[...,0,-2:-1])
                else:
                    pred = self.model(self.rigraph, batch_inputs)
                loss = self.loss_fn(pred, batch_outputs)
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
        start_times_mean = self.stats["start_time"]["mean"]
        start_times_std = self.stats["start_time"]["std"]
        time_diffs_mean = self.stats["time_diffs"]["mean"]
        time_diffs_std = self.stats["time_diffs"]["std"]

        u_mean = torch.tensor(self.stats["u"]["mean"], dtype=self.dtype).to(self.device)
        u_std = torch.tensor(self.stats["u"]["std"], dtype=self.dtype).to(self.device)

        u_in_dim = self.stats["u"]["mean"].shape[0]
        c_in_dim = self.stats["c"]["mean"].shape[0] if "c" in self.stats else 0
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
                if idx == 1 and self.setup_config.visualize_encoder_output:
                    vis_component = self.setup_config.vis_component # [Encoder or Processor]
                    fig = visualize_encoder_output(
                        model =self.model,
                        rigraph=self.rigraph,
                        pndata=x_input,
                        vis_compt = vis_component,
                        channel_in=0,   # 哪个通道可视化 Input
                        domain=[[-1, -1], [1, 1]]
                    )
                    plt.savefig(self.path_config.result_path + f"_{vis_component}_features.png", dpi=300,bbox_inches="tight", pad_inches=0.1)
                    plt.close(fig)
    
                if self.model_config.use_conditional_norm:
                    pred = self.model(self.rigraph, x_input[...,:-1], x_input[...,0,-2:-1])
                else:
                    pred = self.model(self.rigraph, x_input)
                
                if self.dataset_config.stepper_mode == "output":
                    pred_de_norm = pred * u_std + u_mean
                    next_input = pred
                
                elif self.dataset_config.stepper_mode == "residual":
                    res_mean = torch.tensor(self.stats["res"]["mean"], dtype=self.dtype).to(self.device)
                    res_std = torch.tensor(self.stats["res"]["std"], dtype=self.dtype).to(self.device)
                    pred_de_norm = pred * res_std + res_mean

                    u_input_de_norm = current_u_in * u_std + u_mean

                    
                    pred_de_norm = u_input_de_norm + pred_de_norm
                    next_input = (pred_de_norm - u_mean)/u_std
                
                elif self.dataset_config.stepper_mode == "time_der":
                    der_mean = torch.tensor(self.stats["der"]["mean"], dtype=self.dtype).to(self.device)
                    der_std = torch.tensor(self.stats["der"]["std"], dtype=self.dtype).to(self.device)
                    pred_de_norm = pred * der_std + der_mean
                    u_input_de_norm = current_u_in * u_std + u_mean
                    # time difference
                    time_diff_tensor = torch.tensor(time_diff, dtype=self.dtype).to(self.device)
                    pred_de_norm = u_input_de_norm + time_diff_tensor * pred_de_norm
                    next_input = (pred_de_norm - u_mean)/u_std

            # Store prediction
            predictions.append(pred_de_norm)

            # Update current_u_in for next iteration
            current_u_in = next_input
        
        predictions = torch.stack(predictions, dim=1) # Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        
        return predictions
        
    def test(self):
        """
        Evaluate the trained model on the test dataset and optionally plot results.

        Returns:
            None
        """

        self.model.eval()
        self.model.to(self.device)
        self.model.drop_edge = 0.0
        if self.dataset_config.predict_mode == "all":
            modes = ["autoregressive", "direct", "star"]
        else:
            modes = [self.dataset_config.predict_mode]

        errors_dict = {}
        example_data = None # To store for plotting

        for mode in modes:
            all_relative_errors = []
            if mode == "autoregressive":
                time_indices = np.arange(0, 15, 2)  # [0, 2, 4, ..., 14]
            elif mode == "direct":
                time_indices = np.array([0, 14])
            elif mode == "star":
                time_indices = np.array([0, 4, 8, 12, 14])
            elif mode == "fracture":
                time_indices = np.arange(0, 10, 2)
            else:
                raise ValueError(f"Unknown predict_mode: {mode}")
    
            test_dataset = TestDataset(
                u_data = self.test_dataset.u_data,
                c_data = self.test_dataset.c_data,
                t_values = self.test_dataset.t_values,
                metadata = self.metadata,
                time_indices = time_indices,
                stats = self.stats
            ) # x is normalized, y is not normalized

            # TEST = True
            # if TEST:
            #     test_dataset = TestDataset(
            #         u_data = self.train_dataset.u_data,
            #         c_data = self.train_dataset.c_data,
            #         t_values = self.train_dataset.t_values,
            #         metadata = self.metadata,
            #         time_indices = time_indices,
            #         stats = self.stats
            #         )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.test_loader.batch_size,
                shuffle=False,
                num_workers=self.test_loader.num_workers,
                collate_fn=self.collate_fn
            )

            pbar = tqdm(total=len(test_loader), desc=f"Testing ({mode})", colour="blue")
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    # TODO: Figure out whether from CPU to GPU is the compuation bottleneck
                    # x_batch is normalized, y_batch is not normalized
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device) # Shape: [batch_size, num_nodes, num_channels] 
                    pred = self.autoregressive_predict(x_batch, time_indices) # Shape: [batch_size, num_timesteps - 1, num_nodes, num_channels]
                    
                    y_batch_de_norm = y_batch
                    pred_de_norm = pred

                    if self.dataset_config.metric == "final_step":
                        relative_errors = compute_batch_errors(
                            y_batch_de_norm[:,-1,:,:][:,None,:,:], 
                            pred_de_norm[:,-1,:,:][:,None,:,:], 
                            self.metadata)
                    elif self.dataset_config.metric == "all_step":
                        relative_errors = compute_batch_errors(
                            y_batch_de_norm, 
                            pred_de_norm, 
                            self.metadata)
                    else:
                        raise ValueError(f"Unknown metric: {self.dataset_config.metric}")
                    all_relative_errors.append(relative_errors)
                    pbar.update(1)
                    # Store example data for plotting (only once)
                    if example_data is None:
                        u_in_dim = self.stats["u"]["std"].shape[0]
                        if "c" in self.stats:
                            c_in_dim = self.stats["c"]["std"].shape[0]
                            x_batch_input_u = x_batch[...,:u_in_dim].cpu().numpy() * self.stats["u"]["std"] + self.stats["u"]["mean"]
                            x_batch_input_c = x_batch[...,u_in_dim:u_in_dim + c_in_dim].cpu().numpy() * self.stats["c"]["std"] + self.stats["c"]["mean"]
                            x_batch_input = np.stack([x_batch_input_u,x_batch_input_c], axis=-1)
                        else:
                            x_batch_input = x_batch[...,:u_in_dim].cpu().numpy() * self.stats["u"]["std"] + self.stats["u"]["mean"]
                        example_data = {
                            'input': x_batch_input[-1],
                            'coords': self.x_train[0, 0].cpu().numpy(),
                            'gt_sequence': y_batch_de_norm[-1].cpu().numpy(),
                            'pred_sequence': pred_de_norm[-1].cpu().numpy(),
                            'time_indices': time_indices,
                            't_values': self.test_dataset.t_values
                        }

                pbar.close()

            all_relative_errors = torch.cat(all_relative_errors, dim=0)
            final_metric = compute_final_metric(all_relative_errors)
            errors_dict[mode] = final_metric
        print(errors_dict)
        if self.dataset_config.predict_mode == "all":
            self.config.datarow["relative error (direct)"] = errors_dict["direct"]
            self.config.datarow["relative error (auto2)"] = errors_dict["autoregressive"]
            self.config.datarow["relative error (auto4)"] = errors_dict["star"]
        else:
            mode = self.dataset_config.predict_mode
            self.config.datarow[f"relative error ({mode})"] = errors_dict[mode]

        if example_data is not None:
            if self.metadata.names['c']:
                fig = plot_estimates(
                    u_inp = example_data['input'].squeeze(1),
                    u_gtr = np.stack([example_data['gt_sequence'][-1],example_data['input'][...,-1]],axis=-1).squeeze(1),
                    u_prd = np.stack([example_data['pred_sequence'][-1],example_data['input'][...,-1]],axis=-1).squeeze(1),
                    x_inp = example_data['coords'],
                    x_out = example_data['coords'],
                    names = self.metadata.names['u'] + self.metadata.names['c'],
                    symmetric = self.metadata.signed['u'] + self.metadata.signed['c'],
                    domain = self.metadata.domain_x
                )
            
            else:
                fig = plot_estimates(
                    u_inp = example_data['input'],
                    u_gtr = example_data['gt_sequence'][-1],
                    u_prd = example_data['pred_sequence'][-1],
                    x_inp = example_data['coords'],
                    x_out = example_data['coords'],
                    names = self.metadata.names['u'],
                    symmetric = self.metadata.signed['u'],
                    domain = self.metadata.domain_x
                )
            
            fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        
        if self.setup_config.measure_inf_time:
            self.measure_inference_time()

    def plot_results(self, coords, input, gt_sequence, pred_sequence, time_indices, t_values,num_frames=5):
        """
        Plots several frames of ground truth and predicted results using contour plots for all variables.

        Args:
            coords (numpy.ndarray): Coordinates of the nodes. Shape: [num_nodes, num_dims].
            input  (numpy.ndarray): input for the model. Shape: [num_nodes, num_input_channels]
            gt_sequence (numpy.ndarray): Ground truth sequence. Shape: [num_timesteps, num_nodes, num_vars].
            pred_sequence (numpy.ndarray): Predicted sequence. Shape: [num_timesteps, num_nodes, num_vars].
            time_indices (np.ndarray): Array of time indices used in the prediction.
            t_values (np.ndarray): Actual time values corresponding to the time indices.
            num_frames (int): Number of frames to plot. Defaults to 5.

        Saves:
            A plot in the specified results directory, with the file name dynamically including the dataset name.
            For example: ".results/lano/fd/wave_parallel_<dataset_name>.png".
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
                time_idx = time_indices[frame_idx]  # +1 because gt_sequence and pred_sequence start from time_indices[1:]
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
        plt.savefig(self.path_config.result_path)
        plt.close()

    def measure_inference_time(self):
        """
        Measure the inference time of the model and print the average over multiple runs.

        Returns:
            None
        """

        import time
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            # Get a single sample from the test dataset
            x_sample, y_sample = self.test_dataset[0]
            # Ensure inputs are tensors and add batch dimension
            x_sample = torch.tensor(x_sample, dtype=self.dtype).unsqueeze(0).to(self.device)  # Shape: [1, num_nodes, input_dim]
            # Warm-up run
            _ = self.model(self.rigraph, x_sample)
            # Measure inference time over 10 runs
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                if self.model_config.use_conditional_norm:
                    pred = self.model(self.rigraph, x_sample[...,:-1], x_sample[...,0,-2:-1])
                else:
                    pred = self.model(self.rigraph, x_sample)
                # Ensure all CUDA kernels have finished before stopping the timer
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
            print(f"Average inference time over 10 runs (batch size = 1): {avg_time:.6f} seconds")

#######################################
# Foundation Model Trainer
#######################################

class FoundationModelTrainer(TrainerBase):
    """
    Trainer for foundation models using multiple datasets and data parallelism.
    """

    def __init__(self, args):
        """
        Initialize the FoundationModelTrainer.

        Args:
            args (Namespace): Configuration arguments for the trainer.
        """

        super().__init__(args)

    def init_dataset(self, dataset_config):
        """
        Initialize datasets for multiple data sources.

        Args:
            dataset_config (DatasetConfig): Configuration for the datasets, including paths, names, and other settings.

        Returns:
            None
        """

        base_path  = dataset_config.base_path
        dataset_names = dataset_config.names
        metadata_names = dataset_config.metanames
        
        self.datasets = {}
        self.dataloaders = {}
        self.rigraphs = {}
        self.stats = {}
        self.metadata_dict = {}
        self.train_samplers = []

        for dataset_name, metadata_name in zip(dataset_names, metadata_names):
            dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
            metadata = DATASET_METADATA[metadata_name]
            self.metadata_dict[dataset_name] = metadata
            
            with xr.open_dataset(dataset_path) as ds:
                u_array = ds[metadata.group_u].values
                if metadata.group_c is not None:
                    c_array = ds[metadata.group_c].values
                else:
                    c_array = None
                if metadata.group_x is not None:
                    x_array = ds[metadata.group_x].values
                    x_train = x_array
                    self.coord = x_train
                else:
                    domain_x = metadata.domain_x
                    nx, ny = u_array.shape[2], u_array.shape[3]
                    x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                    y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                    xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
                    x_grid = np.stack([xv, yv], axis=-1)
                    x_grid = x_grid.reshape(-1, 2)
                    x_grid = x_grid[None, None, ...]
                    x_train = x_grid
            
            if dataset_name == "wave_c_sines" or dataset_name == "wave_l_sines":
                c_array = torch.full(u_array.shape, 4.0)
    
            active_vars = metadata.active_variables
            u_array = u_array[..., active_vars]

            total_samples = u_array.shape[0]
            train_size = dataset_config.train_size
            val_size = dataset_config.val_size
            test_size = dataset_config.test_size
            assert train_size + val_size + test_size <= total_samples, (
                "Sum of train, val, and test sizes exceeds total samples"
            )

            if dataset_config.rand_dataset:
                indices = np.random.permutation(len(u_array))
            else:
                indices = np.arange(len(u_array))
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size+val_size]
            test_indices = indices[-test_size:]

            u_train = np.ascontiguousarray(u_array[train_indices])
            u_val = np.ascontiguousarray(u_array[val_indices])
            u_test = np.ascontiguousarray(u_array[test_indices])

            if c_array is not None:
                c_train = np.ascontiguousarray(c_array[train_indices])
                c_val = np.ascontiguousarray(c_array[val_indices])
                c_test = np.ascontiguousarray(c_array[test_indices])
            else:
                c_train = c_val = c_test = None
            
            if metadata.domain_t is not None:
                t_start, t_end = metadata.domain_t
                t_values = np.linspace(t_start, t_end, u_array.shape[1])

            else:
                raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")

            max_time_diff = getattr(dataset_config, "max_time_diff", None)
            stats = compute_stats(u_train, c_train, t_values, metadata, max_time_diff,
                                  sample_rate=dataset_config.sample_rate,
                                  use_metadata_stats=dataset_config.use_metadata_stats,
                                  use_time_norm=dataset_config.use_time_norm)
            train_dataset = DynamicPairDataset(u_train, c_train, t_values, metadata,
                                               max_time_diff=max_time_diff,
                                               stepper_mode=dataset_config.stepper_mode,
                                               stats=stats,
                                               use_time_norm=dataset_config.use_time_norm,
                                               dataset_name=dataset_name)
            val_dataset = DynamicPairDataset(u_val, c_val, t_values, metadata,
                                             max_time_diff=max_time_diff,
                                             stepper_mode=dataset_config.stepper_mode,
                                             stats=stats,
                                             use_time_norm=dataset_config.use_time_norm,
                                             dataset_name=dataset_name)
            test_dataset = DynamicPairDataset(u_test, c_test, t_values, metadata,
                                              max_time_diff=max_time_diff,
                                              stepper_mode=dataset_config.stepper_mode,
                                              stats=stats,
                                              use_time_norm=dataset_config.use_time_norm,
                                              dataset_name=dataset_name)
            
            batch_size = dataset_config.batch_size
            shuffle = dataset_config.shuffle
            num_workers = dataset_config.num_workers

            train_sampler = None
            if self.setup_config.distributed:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self.setup_config.world_size,
                    rank=self.setup_config.local_rank
                )
                self.train_samplers.append(train_sampler)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=num_workers,
                collate_fn=self.create_collate_fn(dataset_name),
                sampler = train_sampler
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.create_collate_fn(dataset_name)
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.create_collate_fn(dataset_name)
            )

            self.datasets[dataset_name] = {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset
            }
            self.dataloaders[dataset_name] = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader
            }
            self.stats[dataset_name] = stats
            x_train_tensor = torch.tensor(x_train, dtype = self.dtype).to(self.device)
            rigraph = RegionInteractionGraph.from_point_cloud(
                points = x_train_tensor[0][0],
                phy_domain=self.metadata.domain_x,
                **shallow_asdict(self.graph_config)
            )
            self.rigraphs[dataset_name] = rigraph

        self.train_loader = CombinedDataLoader(
            [self.dataloaders[name]['train'] for name in dataset_names]
        )

        self.val_loader = None

    def create_collate_fn(self, dataset_name):
        """
        Create a custom collate function for each dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            Callable: A collate function that adds dataset name to each batch.
        """

        def collate_fn(batch):
            input_list, output_list = zip(*batch)
            inputs = np.stack(input_list)
            outputs = np.stack(output_list)
            inputs = torch.tensor(inputs, dtype=self.dtype)
            outputs = torch.tensor(outputs, dtype=self.dtype)
            return dataset_name, inputs, outputs
        return collate_fn
    
    def init_graph(self, graph_config):
        """
        Graphs are initialized in init_dataset for each dataset.

        Args:
            graph_config (GraphConfig): Configuration for the graphs.

        Returns:
            None
        """

        pass

    def init_model(self, model_config):
        dataset_name = self.dataset_config.names[0]

        in_channels = self.stats[dataset_name]["u"]["mean"].shape[0] + 2
        
        if model_config.use_conditional_norm:
            in_channels = in_channels - 1
        
        if "c" in self.stats[dataset_name]:
            in_channels += self.stats[dataset_name]["c"]["mean"].shape[0]
        
        out_channels = self.stats[dataset_name]["u"]["mean"].shape[0]

        self.model = init_model_from_rigraph(
            rigraph=self.rigraphs[dataset_name],
            input_size=in_channels,
            output_size=out_channels,
            model=model_config.name,
            drop_edge=model_config.drop_edge,
            variable_mesh=model_config.variable_mesh,
            config=model_config.args
        )

        self.model.to(self.device)
        if self.setup_config.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.setup_config.local_rank],
                output_device=self.setup_config.local_rank
            )
    
    def fit(self, verbose=False):
        result = self.optimizer.optimize(self)
        
        
        # 只在主进程上保存模型
        if self.setup_config.rank == 0:
            self.config.datarow['training time'] = result['time']
            self.save_ckpt()

        if 'train' not in result or len(result['train']['loss']) == 0:
            if self.setup_config.use_variance_test:
                self.variance_test()
            else:
                pass
                self.test()
        else:
            kwargs = {
                "epochs": result['train']['epoch'],
                "losses": result['train']['loss']
            }
        
            if "valid" in result:
                kwargs['val_epochs'] = result['valid']['epoch']
                kwargs['val_losses'] = result['valid']['loss']
            
            if "best" in result:
                kwargs['best_epoch'] = result['best']['epoch']
                kwargs['best_loss'] = result['best']['loss']
            
            if self.setup_config.rank == 0:
                self.plot_losses(**kwargs)

            if self.setup_config.use_variance_test:
                self.variance_test()
            else:
                pass
                self.test()

    def train_step(self, batch):
        self.model.train()
        dataset_name, batch_inputs, batch_outputs = batch
        batch_inputs, batch_outputs = batch_inputs.to(self.device), batch_outputs.to(self.device)
        rigraph = self.rigraphs[dataset_name]

        if self.model_config.use_conditional_norm:
            pred = self.model(rigraph, batch_inputs[..., :-1], batch_inputs[..., 0, -2:-1])
        else:
            pred = self.model(rigraph, batch_inputs)
        loss = self.loss_fn(pred, batch_outputs)
        return loss
    
    def validate(self, loader):
        self.model.eval()
        self.model.drop_edge = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for dataset_name in self.datasets.keys():
                loader = self.dataloaders[dataset_name]["val"]
                rigraph = self.rigraphs[dataset_name]
                stats = self.stats[dataset_name]
                for batch_name, batch_inputs, batch_outputs in loader:
                    batch_inputs, batch_outputs = batch_inputs.to(self.device), batch_outputs.to(self.device)
                    if self.model_config.use_conditional_norm:
                        pred = self.model(rigraph, batch_inputs[..., :-1], batch_inputs[..., 0, -2:-1])
                    else:
                        pred = self.model(rigraph, batch_inputs)
                    loss = self.loss_fn(pred, batch_outputs)
                    total_loss += loss.item()
                total_samples += len(loader)
        return total_loss / total_samples
    
    def autoregressive_predict(self, x_batch, time_indices, rigraph, stats):
        batch_size, num_nodes, input_dim = x_batch.shape
        num_timesteps = len(time_indices)
        predictions = []

        t_values = self.test_t_values 
        start_times_mean = stats["start_time"]["mean"]
        start_times_std = stats["start_time"]["std"]
        time_diffs_mean = stats["time_diffs"]["mean"]
        time_diffs_std = stats["time_diffs"]["std"]

        u_mean = torch.tensor(stats["u"]["mean"], dtype=self.dtype).to(self.device)
        u_std = torch.tensor(stats["u"]["std"], dtype=self.dtype).to(self.device)

        u_in_dim = stats["u"]["mean"].shape[0]
        c_in_dim = stats["c"]["mean"].shape[0] if "c" in stats else 0

        if c_in_dim > 0:
            c_in = x_batch[..., u_in_dim:u_in_dim + c_in_dim]
        else:
            c_in = None

        current_u_in = x_batch[..., :u_in_dim]

        for idx in range(1, num_timesteps):
            t_in_idx = time_indices[idx - 1]
            t_out_idx = time_indices[idx]
            start_time = t_values[t_in_idx]
            time_diff = t_values[t_out_idx] - t_values[t_in_idx]

            start_time_norm = (start_time - start_times_mean) / start_times_std
            time_diff_norm = (time_diff - time_diffs_mean) / time_diffs_std

            start_time_expanded = torch.full((batch_size, num_nodes, 1), start_time_norm, dtype=self.dtype).to(self.device)
            time_diff_expanded = torch.full((batch_size, num_nodes, 1), time_diff_norm, dtype=self.dtype).to(self.device)

            input_features = [current_u_in]
            if c_in is not None:
                input_features.append(c_in)
            input_features.append(start_time_expanded)
            input_features.append(time_diff_expanded)
            x_input = torch.cat(input_features, dim=-1)

            with torch.no_grad():
                if self.model_config.use_conditional_norm:
                    pred = self.model(rigraph, x_input[..., :-1], x_input[..., 0, -2:-1])
                else:
                    pred = self.model(rigraph, x_input)

                if self.dataset_config.stepper_mode == "output":
                    pred_de_norm = pred * u_std + u_mean
                    next_input = pred
                
                elif self.dataset_config.stepper_mode == "residual":
                    res_mean = torch.tensor(stats["res"]["mean"], dtype=self.dtype).to(self.device)
                    res_std = torch.tensor(stats["res"]["std"], dtype=self.dtype).to(self.device)
                    pred_de_norm = pred * res_std + res_mean
                    u_input_de_norm = current_u_in * u_std + u_mean
                    pred_de_norm = u_input_de_norm + pred_de_norm
                    next_input = (pred_de_norm - u_mean) / u_std

                elif self.dataset_config.stepper_mode == "time_der":
                    der_mean = torch.tensor(stats["der"]["mean"], dtype=self.dtype).to(self.device)
                    der_std = torch.tensor(stats["der"]["std"], dtype=self.dtype).to(self.device)
                    pred_de_norm = pred * der_std + der_mean
                    u_input_de_norm = current_u_in * u_std + u_mean
                    time_diff_tensor = torch.tensor(time_diff, dtype=self.dtype).to(self.device)
                    pred_de_norm = u_input_de_norm + time_diff_tensor * pred_de_norm
                    next_input = (pred_de_norm - u_mean) / u_std

            predictions.append(pred_de_norm)
            current_u_in = next_input

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        self.model.drop_edge = 0.0

        if self.dataset_config.predict_mode == "all":
            modes = ["autoregressive", "direct", "star"]
        else:
            modes = [self.dataset_config.predict_mode]

        for dataset_name in self.datasets.keys():
            errors_dict = {}
            example_data = None  

            rigraph = self.rigraphs[dataset_name]
            stats = self.stats[dataset_name]
            test_dataset = self.datasets[dataset_name]["test"]
            metadata = self.metadata_dict[dataset_name]
            t_values = test_dataset.t_values
            self.test_t_values = t_values 

            for mode in modes:
                all_relative_errors = []
                if mode == "autoregressive":
                    time_indices = np.arange(0, 15, 2)
                elif mode == "direct":
                    time_indices = np.array([0, 14])
                elif mode == "star":
                    time_indices = np.array([0, 4, 8, 12, 14])
                else:
                    raise ValueError(f"Unknown predict_mode: {mode}")

                test_data = TestDataset(
                    u_data=test_dataset.u_data,
                    c_data=test_dataset.c_data,
                    t_values=test_dataset.t_values,
                    metadata=metadata,
                    time_indices=time_indices,
                    stats=stats
                )
                test_loader = DataLoader(
                    test_data,
                    batch_size=self.dataloaders[dataset_name]["test"].batch_size,
                    shuffle=False,
                    num_workers=self.dataloaders[dataset_name]["test"].num_workers,
                    collate_fn=self.create_collate_fn(dataset_name)
                )

                if self.setup_config.rank == 0:
                    pbar = tqdm(total=len(test_loader), desc=f"Testing {dataset_name} ({mode})", colour="blue")
                else:
                    pbar = None
                
                with torch.no_grad():
                    for i, (dataset_names, x_batch, y_batch) in enumerate(test_loader):
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        pred = self.autoregressive_predict(x_batch, time_indices, rigraph, stats)
                        y_batch_de_norm = y_batch
                        pred_de_norm = pred

                        if self.dataset_config.metric == "final_step":
                            relative_errors = compute_batch_errors(
                                y_batch_de_norm[:, -1, :, :][:, None, :, :],
                                pred_de_norm[:, -1, :, :][:, None, :, :],
                                metadata
                            )
                        elif self.dataset_config.metric == "all_step":
                            relative_errors = compute_batch_errors(
                                y_batch_de_norm,
                                pred_de_norm,
                                metadata
                            )
                        else:
                            raise ValueError(f"Unknown metric: {self.dataset_config.metric}")
                        all_relative_errors.append(relative_errors)
                        
                        if pbar is not None:
                            pbar.update(1)

                        if example_data is None:
                            u_in_dim = stats["u"]["std"].shape[0]
                        if "c" in self.stats:
                            c_in_dim = self.stats["c"]["std"].shape[0]
                            x_batch_input_u = x_batch[...,:u_in_dim].cpu().numpy() * self.stats["u"]["std"] + self.stats["u"]["mean"]
                            x_batch_input_c = x_batch[...,u_in_dim:u_in_dim + c_in_dim].cpu().numpy() * self.stats["c"]["std"] + self.stats["c"]["mean"]
                            x_batch_input = np.stack([x_batch_input_u,x_batch_input_c], axis=-1)
                        else:
                            x_batch_input = x_batch[...,:u_in_dim].cpu().numpy() * stats["u"]["std"] + stats["u"]["mean"]
                            example_data = {
                                'input': x_batch_input[-1],
                                'coords': self.rigraphs[dataset_name].physical_to_regional.src_ndata['pos'].cpu().numpy(),
                                'gt_sequence': y_batch_de_norm[-1].cpu().numpy(),
                                'pred_sequence': pred_de_norm[-1].cpu().numpy(),
                                'time_indices': time_indices,
                                't_values': test_dataset.t_values
                            }
                    
                    if pbar is not None:
                        pbar.close()


                all_relative_errors = torch.cat(all_relative_errors, dim=0)
                final_metric = compute_final_metric(all_relative_errors)
                errors_dict[mode] = final_metric
            if self.setup_config.rank == 0:
                print(f"Results for {dataset_name}: {errors_dict}")

                self.config.datarow[f"{dataset_name} relative error"] = errors_dict
                # if example_data is not None:
                #     self.plot_results(
                #         coords=example_data['coords'],
                #         input = example_data['input'],
                #         gt_sequence=example_data['gt_sequence'],
                #         pred_sequence=example_data['pred_sequence'],
                #         time_indices=example_data['time_indices'],
                #         t_values=example_data['t_values'],
                #         dataset_name=dataset_name,
                #         num_frames=example_data['gt_sequence'].shape[0],
                #     )
                
                if example_data is not None:
                    if metadata.names['c']:
                        fig = plot_estimates(
                            u_inp = example_data['input'].squeeze(1),
                            u_gtr = np.stack([example_data['gt_sequence'][-1],example_data['input'][...,-1]],axis=-1).squeeze(1),
                            u_prd = np.stack([example_data['pred_sequence'][-1],example_data['input'][...,-1]],axis=-1).squeeze(1),
                            x_inp = self.coord[0][0],
                            x_out = self.coord[0][0],
                            names = metadata.names['u'] + metadata.names['c'],
                            symmetric = metadata.signed['u'] + metadata.signed['c'],
                            domain = metadata.domain_x
                        )
                    
                    else:
                        fig = plot_estimates(
                            u_inp = example_data['input'],
                            u_gtr = example_data['gt_sequence'][-1],
                            u_prd = example_data['pred_sequence'][-1],
                            x_inp = self.coord[0][0],
                            x_out = self.coord[0][0],
                            names = metadata.names['u'],
                            symmetric = metadata.signed['u'],
                            domain = metadata.domain_x
                        )
                    
                    fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
                    plt.close(fig)
            
    def plot_results(self, coords, input, gt_sequence, pred_sequence, time_indices, t_values, dataset_name, num_frames=5):
        """
        Plots several frames of ground truth and predicted results using contour plots for all variables.

        Args:
            coords (numpy.ndarray): Coordinates of the nodes. Shape: [num_nodes, num_dims].
            input  (numpy.ndarray): input for the model. Shape: [num_nodes, num_input_channels]
            gt_sequence (numpy.ndarray): Ground truth sequence. Shape: [num_timesteps, num_nodes, num_vars].
            pred_sequence (numpy.ndarray): Predicted sequence. Shape: [num_timesteps, num_nodes, num_vars].
            time_indices (np.ndarray): Array of time indices used in the prediction.
            t_values (np.ndarray): Actual time values corresponding to the time indices.
            dataset_name (str): Name of the dataset, which will be included in the saved file name.
            num_frames (int): Number of frames to plot. Defaults to 5.

        Saves:
            A plot in the specified results directory, with the file name dynamically including the dataset name.
            For example: ".results/lano/fd/wave_parallel_<dataset_name>.png".
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
                time_idx = time_indices[frame_idx]  # +1 because gt_sequence and pred_sequence start from time_indices[1:]
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
        base_path, ext = os.path.splitext(self.path_config.result_path)
        new_save_path = f"{base_path}_{dataset_name}{ext}"
        plt.savefig(new_save_path)
        plt.close()
