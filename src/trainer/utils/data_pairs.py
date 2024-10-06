import numpy as np
from ...data.dataset import Metadata
from torch.utils.data import Dataset
import torch

def create_all_to_all_pairs(u_data: np.array , x_data: np.array, c_data: np.array, metadata: Metadata):
    """
        This code is mainly used for time-dependent PDE, for constructing the all-to-all pairs dataset.
    """
    num_samples, num_timesteps, num_nodes, num_vars = u_data.shape

    if metadata.domain_t is not None:
        t_start, t_end = metadata.domain_t  # For example, (0.0, 1.0)
    else:
        raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")

    # Compute actual time values
    time_values = np.linspace(t_start, t_end, num_timesteps)  # Shape: [num_timesteps]

    # Create all possible combinations of t_in and t_out where t_out > t_in
    t_in_indices, t_out_indices = np.meshgrid(np.arange(num_timesteps), np.arange(num_timesteps), indexing='ij')
    t_in_indices = t_in_indices.flatten()  # Shape: [num_timesteps^2]
    t_out_indices = t_out_indices.flatten()

    # Only keep pairs where t_out > t_in, but it can be optimized with max_time_dff
    # for example:
    # max_time_diff = 5
    # time_diffs = t_out_indices - t_in_indices
    # mask = (t_out_indices > t_in_indices) & (time_diffs <= max_time_diff)
    mask = t_out_indices > t_in_indices
    t_in_indices = t_in_indices[mask]
    t_out_indices = t_out_indices[mask]

    num_pairs = t_in_indices.shape[0]
    assert num_timesteps * (num_timesteps - 1) // 2 == num_pairs, f"Expected {num_timesteps * (num_timesteps - 1) // 2} pairs for every data, but found {num_pairs}."

    # Expand sample indices
    sample_indices = np.repeat(np.arange(num_samples), num_pairs)  # Shape: [num_samples * num_pairs]

    # Repeat t_in_indices and t_out_indices for all samples
    t_in_indices_expanded = np.tile(t_in_indices, num_samples)  # Shape: [num_samples * num_pairs]
    t_out_indices_expanded = np.tile(t_out_indices, num_samples)

    # Get u_in and u_out
    u_in = u_data[sample_indices, t_in_indices_expanded, :, :]  # Shape: [num_samples * num_pairs, num_nodes, num_vars]
    u_out = u_data[sample_indices, t_out_indices_expanded, :, :]  # Same shape

    # Get x_in and x_out
    x_in = x_data[sample_indices, t_in_indices_expanded, :, :]  # Shape: [num_samples * num_pairs, num_nodes, num_dims]
    x_out = x_data[sample_indices, t_out_indices_expanded, :, :]  # Same shape

    # Map time indices to actual time values
    t_in_times = time_values[t_in_indices_expanded]  # Shape: [num_samples * num_pairs]
    t_out_times = time_values[t_out_indices_expanded]  # Same shape

    # Compute lead_times and time_diffs in actual time units
    lead_times = t_out_times[:, np.newaxis]  # Shape: [num_samples * num_pairs, 1]
    time_diffs = (t_out_times - t_in_times)[:, np.newaxis]  # Shape: [num_samples * num_pairs, 1]

    if c_data is not None:
        c_in = c_data[sample_indices, t_in_indices_expanded, :, :]  # Shape: [num_samples * num_pairs, num_nodes, num_c_vars]
    else:
        c_in = None

    return u_in, u_out, x_in, x_out, lead_times, time_diffs, c_in

class DynamicPairDataset_half(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, max_time_diff=14, time_stats=None):
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

        # Only use the first 15 time steps
        self.num_timesteps = max_time_diff
        self.t_values = self.t_values[:self.num_timesteps + 1] # Shape: [num_timesteps + 1]

        # Generate specific time pairs
        self.t_in_indices = []
        self.t_out_indices = []
        for lag in range(2, self.num_timesteps + 1, 2):  # Even lags from 2 to 14
            num_pairs = (self.num_timesteps - lag) // 2 + 1
            for i in range(0, self.num_timesteps - lag + 1, 2):
                t_in_idx = i
                t_out_idx = i + lag
                self.t_in_indices.append(t_in_idx)
                self.t_out_indices.append(t_out_idx)
        
        # Convert lists to numpy arrays
        self.t_in_indices = np.array(self.t_in_indices)
        self.t_out_indices = np.array(self.t_out_indices)

        # Time differences
        self.time_diffs = self.t_values[self.t_out_indices] - self.t_values[self.t_in_indices] # Shape: [num_time_pairs]

        # Total number of time pairs per sample
        self.num_time_pairs = len(self.t_in_indices)

        # Total number of samples in the dataset
        self.total_pairs = self.num_samples * self.num_time_pairs

        # Compute time features for all pairs
        self.start_times = self.t_values[self.t_in_indices] # Shape: [num_time_pairs]

        # Compute time statistics if not provided
        if time_stats is None:
            self.start_times_mean = np.mean(self.start_times)
            self.start_times_std = np.std(self.start_times) + 1e-10
            self.time_diffs_mean = np.mean(self.time_diffs)
            self.time_diffs_std = np.std(self.time_diffs) + 1e-10
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

        # pre-expand normalized time features to match num_nodes
        self.start_time_expanded = np.tile(self.start_times_norm[:, np.newaxis], (1, self.num_nodes)) # Shape: [num_time_pairs, num_nodes]
        self.time_diff_expanded = np.tile(self.time_diffs_norm[:, np.newaxis], (1, self.num_nodes)) # Shape: [num_time_pairs, num_nodes]

        # Reshape to [num_time_pairs, num_nodes, 1]
        self.start_time_expanded = self.start_time_expanded[..., np.newaxis]
        self.time_diff_expanded = self.time_diff_expanded[..., np.newaxis]

        if self.c_data is not None:
            self.c_in_data = self.c_data[:, :,self.num_timesteps]
        else:
            self.c_in_data = None

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

class DynamicPairDataset_half_batch(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, max_time_diff=14, time_stats=None):
        self.u_data = u_data  # 已经是GPU上的张量
        self.c_data = c_data  # 已经是GPU上的张量或None
        self.t_values = t_values  # 已经是GPU上的张量
        self.metadata = metadata
        self.num_samples, self.num_timesteps, self.num_nodes, self.num_vars = u_data.shape

        # 只使用前max_time_diff个时间步
        self.num_timesteps = max_time_diff
        self.t_values = self.t_values[:self.num_timesteps + 1]

        # 生成时间对
        self.t_in_indices = []
        self.t_out_indices = []
        for lag in range(2, self.num_timesteps + 1, 2):
            for i in range(0, self.num_timesteps - lag + 1, 2):
                t_in_idx = i
                t_out_idx = i + lag
                self.t_in_indices.append(t_in_idx)
                self.t_out_indices.append(t_out_idx)

        # 转换为GPU上的张量
        self.t_in_indices = torch.tensor(self.t_in_indices, dtype=torch.long, device=u_data.device)
        self.t_out_indices = torch.tensor(self.t_out_indices, dtype=torch.long, device=u_data.device)

        # 计算时间差
        self.time_diffs = self.t_values[self.t_out_indices] - self.t_values[self.t_in_indices]

        # 总的时间对数量
        self.num_time_pairs = len(self.t_in_indices)
        self.total_pairs = self.num_samples * self.num_time_pairs

        # 计算时间特征
        self.start_times = self.t_values[self.t_in_indices]

        # 计算时间统计量
        if time_stats is None:
            self.start_times_mean = self.start_times.mean()
            self.start_times_std = self.start_times.std() + 1e-10
            self.time_diffs_mean = self.time_diffs.mean()
            self.time_diffs_std = self.time_diffs.std() + 1e-10
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

        # 预计算归一化的时间特征
        self.start_times_norm = (self.start_times - self.start_times_mean) / self.start_times_std
        self.time_diffs_norm = (self.time_diffs - self.time_diffs_mean) / self.time_diffs_std

        # 扩展时间特征以匹配节点数量
        self.start_time_expanded = self.start_times_norm.unsqueeze(-1).unsqueeze(-1)  # [num_time_pairs, 1, 1]
        self.time_diff_expanded = self.time_diffs_norm.unsqueeze(-1).unsqueeze(-1)    # [num_time_pairs, 1, 1]

        if self.c_data is not None:
            self.c_in_data = self.c_data[:, :, :self.num_timesteps]
        else:
            self.c_in_data = None

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        sample_idx = idx // self.num_time_pairs
        time_pair_idx = idx % self.num_time_pairs

        t_in_idx = self.t_in_indices[time_pair_idx]
        t_out_idx = self.t_out_indices[time_pair_idx]

        # 获取输入和输出数据
        u_in = self.u_data[sample_idx, t_in_idx]      # [num_nodes, num_vars]
        u_out = self.u_data[sample_idx, t_out_idx]    # [num_nodes, num_vars]

        # 获取时间特征
        start_time_expanded = self.start_time_expanded[time_pair_idx]  # [1, 1]
        time_diff_expanded = self.time_diff_expanded[time_pair_idx]    # [1, 1]

        # 如果c_data可用
        if self.c_data is not None:
            c_in = self.c_in_data[sample_idx, t_in_idx]
        else:
            c_in = None

        # 准备输入特征
        input_features = [u_in]
        if c_in is not None:
            input_features.append(c_in)
        input_features = torch.cat(input_features, dim=-1)

        num_nodes = u_in.shape[0]
        # 扩展时间特征以匹配节点数量
        start_time_expanded = start_time_expanded.expand(num_nodes, 1)
        time_diff_expanded = time_diff_expanded.expand(num_nodes, 1)

        # 合并所有输入特征
        input_features = torch.cat([input_features, start_time_expanded, time_diff_expanded], dim=-1)

        return input_features, u_out


class DynamicPairDataset_full(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, max_time_diff=None, time_stats=None):
        """
        Custom Dataset that generates data pairs on-the-fly using index pairs.
        Args:
            u_data (numpy.ndarray): Shape [num_samples, num_timesteps, num_nodes, num_vars]
            c_data (numpy.ndarray): Shape [num_samples, num_timesteps, num_nodes, num_c_vars] or None
            t_values (numpy.ndarray): Actual time values corresponding to timesteps
            metadata (Metadata): Metadata object containing domain information
            max_time_diff (float): Maximum allowed time difference between t_out and t_in
            time_stats (dict): Dictionary containing time statistics
        """
        self.u_data = u_data
        self.c_data = c_data
        self.t_values = t_values  # Shape: [num_timesteps]
        self.metadata = metadata
        self.num_samples, self.num_timesteps, self.num_nodes, self.num_vars = u_data.shape
        self.max_time_diff = max_time_diff

        # Generate index pairs (t_in, t_out) where t_out > t_in
        t_in_indices, t_out_indices = np.meshgrid(np.arange(self.num_timesteps), np.arange(self.num_timesteps), indexing='ij')
        t_in_indices = t_in_indices.flatten()
        t_out_indices = t_out_indices.flatten()
        mask = t_out_indices > t_in_indices

        # Apply max_time_diff constraint if provided
        time_diffs = self.t_values[t_out_indices] - self.t_values[t_in_indices]
        if max_time_diff is not None:
            mask &= (time_diffs <= max_time_diff)

        self.t_in_indices = t_in_indices[mask]
        self.t_out_indices = t_out_indices[mask]
        self.time_diffs = time_diffs[mask]

        # Total number of time pairs per sample
        self.num_time_pairs = len(self.t_in_indices)

        # Total number of samples in the dataset
        self.total_pairs = self.num_samples * self.num_time_pairs

        self.lead_times = self.t_values[self.t_in_indices]

        if time_stats is None:
            self.lead_times_mean = np.mean(self.lead_times)
            self.lead_times_std = np.std(self.lead_times) + 1e-10
            self.time_diffs_mean = np.mean(self.time_diffs)
            self.time_diffs_std = np.std(self.time_diffs) + 1e-10
            self.time_stats = {
                'lead_times_mean': self.lead_times_mean,
                'lead_times_std': self.lead_times_std,
                'time_diffs_mean': self.time_diffs_mean,
                'time_diffs_std': self.time_diffs_std
            }
        else:
            self.time_stats = time_stats
            self.lead_times_mean = time_stats['lead_times_mean']
            self.lead_times_std = time_stats['lead_times_std']
            self.time_diffs_mean = time_stats['time_diffs_mean']
            self.time_diffs_std = time_stats['time_diffs_std']

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        # Compute sample index and time pair index
        sample_idx = idx // self.num_time_pairs
        time_pair_idx = idx % self.num_time_pairs

        t_in_idx = self.t_in_indices[time_pair_idx]
        t_out_idx = self.t_out_indices[time_pair_idx]

        # Fetch data for the given indices
        u_in = self.u_data[sample_idx, t_in_idx]  # Shape: [num_nodes, num_vars]
        u_out = self.u_data[sample_idx, t_out_idx]  # Shape: [num_nodes, num_vars]

        # Compute time features
        lead_time = self.lead_times[time_pair_idx]
        time_diff = self.time_diffs[time_pair_idx]

        lead_time_norm = (lead_time - self.lead_times_mean) / self.lead_times_std
        time_diff_norm = (time_diff - self.time_diffs_mean) / self.time_diffs_std

        # If c_data is available
        if self.c_data is not None:
            c_in = self.c_data[sample_idx, t_in_idx]  # Shape: [num_nodes, num_c_vars]
        else:
            c_in = None

        # Prepare input features
        input_features = [u_in]
        if c_in is not None:
            input_features.append(c_in)
        input_features = np.concatenate(input_features, axis=-1)  # Shape: [num_nodes, input_dim]

        # Add time features (expanded to match num_nodes)
        lead_time_expanded = np.full((self.num_nodes, 1), lead_time_norm)
        time_diff_expanded = np.full((self.num_nodes, 1), time_diff_norm)
        input_features = np.concatenate([input_features, lead_time_expanded, time_diff_expanded], axis=-1)

        return input_features, u_out  # Both are NumPy arrays

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