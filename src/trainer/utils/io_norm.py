import numpy as np

def compute_stats(u_data,c_data,t_values, max_time_diff = 14, use_metadata_stats = False, sample_rate = 0.1):
    """

    """
    self.stats = {}
    self.stats["u"] = {}
    self.stats["res"] = {}
    self.stats["der"] = {}

    self.stats["u"]["mean"] = np.mean(u_data, axis=(0,1,2))  # Shape: [num_active_vars]
    self.stats["u"]["std"] = np.std(u_data, axis=(0,1,2)) + EPSILON
    if use_metadata_stats:
        self.stats["u"]["mean"] = self.metadata.global_mean
        self.stats["u"]["std"] = self.metadata.global_std
    if c_data is not None:
        self.stats["c"] = {}
        self.stats["c"]["mean"] = np.mean(c_data, axis=(0,1,2))
        self.stats["c"]["std"] = np.std(c_data, axis=(0,1,2))
    
    

