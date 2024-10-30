import numpy as np
from ...data.dataset import Metadata
from torch.utils.data import Dataset
import torch

class DynamicPairDataset(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, max_time_diff=14, time_stats=None, use_time_norm=True):
        """
        Custom Dataset that generates specific time pairs for training.

        Args:
            u_data (numpy.ndarray): Shape [num_samples, num_timesteps, num_nodes, num_vars]
            c_data (numpy.ndarray): Shape: [num_samples, num_timesteps, num_nodes, num_c_vars] or None
            t_values (numpy.ndarray): Actual time values corresponding to timesteps
            metadata (Metadata): Metadata object containing domain information
            max_time_diff (int): Maximum allowed time difference between t_out and t_in
            time_stats (dict): Dictionary to store time statistics
        """
        self.u_data = u_data
        self.c_data = c_data
        self.t_values = t_values  # Shape: [num_timesteps]
        self.metadata = metadata
        self.num_samples, self.num_timesteps, self.num_nodes, self.num_vars = u_data.shape

        # if max_time_diff = 14, Only use the first 15 time steps
        self.num_timesteps = max_time_diff
        self.t_values = self.t_values[:self.num_timesteps + 1] # Shape: [num_timesteps + 1]

        # Generate specific time pairs using index
        self.t_in_indices = []
        self.t_out_indices = []
        for lag in range(2, self.num_timesteps + 1, 2):  # Even lags from 2 to 14
            num_pairs = (self.num_timesteps - lag) // 2 + 1
            for i in range(0, self.num_timesteps - lag + 1, 2):
                t_in_idx = i
                t_out_idx = i + lag
                self.t_in_indices.append(t_in_idx)
                self.t_out_indices.append(t_out_idx)

        self.t_in_indices = np.array(self.t_in_indices)
        self.t_out_indices = np.array(self.t_out_indices)

        self.time_diffs = self.t_values[self.t_out_indices] - self.t_values[self.t_in_indices] # Shape: [num_time_pairs]
        self.num_time_pairs = len(self.t_in_indices)
        self.total_pairs = self.num_samples * self.num_time_pairs # The total size of the training data pairs.

        self.start_times = self.t_values[self.t_in_indices] # Shape: [num_time_pairs]

        # Compute time statistics if not provided
        if time_stats is None:
            self.start_times_mean = np.mean(self.start_times)
            self.start_times_std = np.std(self.start_times) + 1e-10
            self.time_diffs_mean = np.mean(self.time_diffs)
            self.time_diffs_std = np.std(self.time_diffs) + 1e-10
            if not use_time_norm:
                self.start_times_mean = 0.0
                self.start_times_std = 1.0
                self.time_diffs_mean = 0.0
                self.time_diffs_std = 1.0

            self.time_stats = {
                'start_times_mean': self.start_times_mean,
                'start_times_std': self.start_times_std,
                'time_diffs_mean': self.time_diffs_mean,
                'time_diffs_std': self.time_diffs_std,
            }
        
        else:
            self.time_stats = time_stats
            self.start_times_mean = time_stats['start_times_mean']
            self.start_times_std = time_stats['start_times_std']
            self.time_diffs_mean = time_stats['time_diffs_mean']
            self.time_diffs_std = time_stats['time_diffs_std']
        
        # precompute the normalized time features for all time pairs
        self.start_times_norm = (self.start_times - self.start_times_mean) / self.start_times_std
        self.time_diffs_norm = (self.time_diffs - self.time_diffs_mean) / self.time_diffs_std

        self.start_time_expanded = np.tile(self.start_times_norm[:, np.newaxis], (1, self.num_nodes)) # Shape: [num_time_pairs, num_nodes]
        self.time_diff_expanded = np.tile(self.time_diffs_norm[:, np.newaxis], (1, self.num_nodes)) # Shape: [num_time_pairs, num_nodes]

        # Reshape to [num_time_pairs, num_nodes, 1]
        self.start_time_expanded = self.start_time_expanded[..., np.newaxis]
        self.time_diff_expanded = self.time_diff_expanded[..., np.newaxis]

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        # Compute sample index and time pair index
        sample_idx = idx // self.num_time_pairs
        time_pair_idx = idx % self.num_time_pairs

        t_in_idx = self.t_in_indices[time_pair_idx]
        t_out_idx = self.t_out_indices[time_pair_idx]

        # Fetch data for the given indices
        u_in = self.u_data[sample_idx, t_in_idx]  # Input at t_in, Shape: [num_nodes, num_vars]
        u_out = self.u_data[sample_idx, t_out_idx]  # Output at t_out, Shape: [num_nodes, num_vars]

        # Fetch time features
        start_time_expanded = self.start_time_expanded[time_pair_idx] # Shape: [num_nodes, 1]
        time_diff_expanded = self.time_diff_expanded[time_pair_idx] # Shape: [num_nodes, 1]

        # If c_data is available
        if self.c_data is not None:
            c_in = self.c_data[sample_idx, t_in_idx]
        else:
            c_in = None

        # Prepare input features
        input_features = [u_in]
        if c_in is not None:
            input_features.append(c_in)
        input_features = np.concatenate(input_features, axis=-1)

        # Add normalized time features (expanded to match num_nodes)
        input_features = np.concatenate([input_features, start_time_expanded, time_diff_expanded], axis=-1)

        return input_features, u_out

class TestDataset(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, time_indices):
        """
        Custom dataset for testing, providing initial input and ground truth sequences.
        
        Args:
            u_data (numpy.ndarray): Shape [num_samples, num_timesteps, num_nodes, num_vars]
            c_data (numpy.ndarray or None): Shape [num_samples, num_timesteps, num_nodes, num_c_vars]
            t_values (numpy.ndarray): Actual time values corresponding to timesteps
            metadata (Metadata): Metadata object containing domain information
            time_indices (list or np.ndarray): Time indices to consider (e.g., [0, 2, 4, ..., 14])
        """
        self.u_data = u_data
        self.c_data = c_data
        self.t_values = t_values
        self.metadata = metadata
        self.time_indices = time_indices
        self.num_samples = u_data.shape[0]
        self.num_nodes = u_data.shape[2]
        self.num_vars = u_data.shape[3]
        self.dtype = np.float32  # or np.float64, depending on your data type
        
        # Precompute normalized u_data if necessary (using self.u_mean and self.u_std)
        # Assuming self.u_mean and self.u_std are already computed and available

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get initial input at time t=0
        u_in = self.u_data[idx, self.time_indices[0]]  # Shape: [num_nodes, num_vars]
        # Get ground truth outputs at future time steps
        y_sequence = self.u_data[idx, self.time_indices[1:]]  # Shape: [num_timesteps - 1, num_nodes, num_vars]
        
        # If c_data is available
        if self.c_data is not None:
            c_in = self.c_data[idx, self.time_indices[0]]  # Shape: [num_nodes, num_c_vars]
            # Combine u_in and c_in
            input_features = np.concatenate([u_in, c_in], axis=-1)
        else:
            input_features = u_in  # Shape: [num_nodes, num_vars]
        
        # Note: Time features will be added in the `autoregressive_predict` function
        return input_features.astype(self.dtype), y_sequence.astype(self.dtype)