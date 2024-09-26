import numpy as np
import matplotlib.pyplot as plt 

def plot_results(self, coords, gt_sequence, pred_sequence, time_indices, t_values, num_frames=5):
    """
    Plots several frames of ground truth and predicted results using contour plots.

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
    
    # Create a figure with 3 rows and num_frames columns
    fig, axes = plt.subplots(3, num_frames, figsize=(4 * num_frames, 12))
    
    vmin = min(gt_sequence.min(), pred_sequence.min())
    vmax = max(gt_sequence.max(), pred_sequence.max())
    
    for i, frame_idx in enumerate(frame_indices):
        time_idx = time_indices[frame_idx + 1]  # +1 because gt_sequence and pred_sequence start from time_indices[1:]
        time_value = t_values[time_idx]
        
        gt = gt_sequence[frame_idx][:, 0]  # Assuming single variable
        pred = pred_sequence[frame_idx][:, 0]
        abs_error = np.abs(gt - pred)
        
        # Ground Truth
        ax_gt = axes[0, i]
        ct_gt = ax_gt.tricontourf(x, y, gt, cmap='plasma')
        ax_gt.set_title(f"Time: {time_value:.2f}")
        if i == 0:
            ax_gt.set_ylabel('Ground Truth')
        ax_gt.set_aspect('equal')
        plt.colorbar(ct_gt, ax=ax_gt)
        
        # Prediction
        ax_pred = axes[1, i]
        ct_pred = ax_pred.tricontourf(x, y, pred, cmap='plasma')
        if i == 0:
            ax_pred.set_ylabel('Prediction')
        ax_pred.set_aspect('equal')
        plt.colorbar(ct_pred, ax=ax_pred)
        
        # Absolute Error
        ax_error = axes[2, i]
        ct_error = ax_error.tricontourf(x, y, abs_error, cmap='plasma')
        if i == 0:
            ax_error.set_ylabel('Absolute Error')
        ax_error.set_aspect('equal')
        plt.colorbar(ct_error, ax=ax_error)
        
    plt.tight_layout()
    plt.savefig(self.path_config["result_path"])
    plt.close()
