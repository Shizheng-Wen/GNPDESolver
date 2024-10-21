import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os


datasets_to_check = [
    "CE-Gauss",
    "CE-RP",
    "NS-Gauss",
    "NS-SVS",
    "NS-PwC",
    "NS-SL"
]


base_path = "/cluster/work/math/camlab-data/rigno-unstructured/"

num_datasets = len(datasets_to_check)
fig_means, axes_means = plt.subplots(2, 3, figsize=(18, 10))
fig_stds, axes_stds = plt.subplots(2, 3, figsize=(18, 10))

axes_means = axes_means.flatten()
axes_stds = axes_stds.flatten()

for idx, dataset_name in enumerate(datasets_to_check):
    print(f"Examing the dataset：{dataset_name}")
    dataset_path = os.path.join(base_path, dataset_name + ".nc")
    ds = xr.open_dataset(dataset_path)
    
    u = ds['u'].values  # Shape：(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3)
    
    total_samples = u.shape[0]  # phony_dim_0
    train_indices = np.arange(0, 1024)
    val_indices = np.arange(1024, 1024 + 128)
    test_indices = np.arange(1024 + 128, 1024 + 128 + 256)
    

    u_train = u[train_indices]
    u_val = u[val_indices]
    

    u_train_flat = u_train.reshape(u_train.shape[0], -1)  # (num_train_samples, features)
    u_val_flat = u_val.reshape(u_val.shape[0], -1)
    
    u_train_means = np.mean(u_train_flat, axis=1)
    u_val_means = np.mean(u_val_flat, axis=1)
    
    u_train_stds = np.std(u_train_flat, axis=1)
    u_val_stds = np.std(u_val_flat, axis=1)
    
    # Kolmogorov-Smirnov testing
    mean_stat, mean_pvalue = ks_2samp(u_train_means, u_val_means)
    std_stat, std_pvalue = ks_2samp(u_train_stds, u_val_stds)
    
    # Plot mean histogram
    ax_means = axes_means[idx]
    ax_means.hist(u_train_means, bins=50, alpha=0.5, label='Training Set Means')
    ax_means.hist(u_val_means, bins=50, alpha=0.5, label='Validation Set Means')
    ax_means.legend()
    ax_means.set_title(f'{dataset_name} - Mean Comparison\nKS Test p-value={mean_pvalue:.3e}')
    ax_means.set_xlabel('Mean')
    ax_means.set_ylabel('Frequency')
    
    # Plot standard deviation histograms
    ax_stds = axes_stds[idx]
    ax_stds.hist(u_train_stds, bins=50, alpha=0.5, label='Training Set Std Dev')
    ax_stds.hist(u_val_stds, bins=50, alpha=0.5, label='Validation Set Std Dev')
    ax_stds.legend()
    ax_stds.set_title(f'{dataset_name} - Std Dev Comparison\nKS Test p-value={std_pvalue:.3e}')
    ax_stds.set_xlabel('Standard Deviation')
    ax_stds.set_ylabel('Frequency')
    
    print(f"{dataset_name} - Mean KS Test: Statistic={mean_stat}, p-value={mean_pvalue}")
    print(f"{dataset_name} - Std Dev KS Test: Statistic={std_stat}, p-value={std_pvalue}")

fig_means.tight_layout()
fig_stds.tight_layout()

fig_means.savefig('means_comparison.png')
fig_stds.savefig('stds_comparison.png')
