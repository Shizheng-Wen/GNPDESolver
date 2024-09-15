import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.data import Data, Dataset

import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
from architectures.gino import GINO
from architectures.utils import count_model_params

class PoissonDataset(Dataset):
    def __init__(self, c, u, x):
        self.c = c
        self.u = u
        self.x = x

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        c_sample = torch.tensor(self.c[idx].values, dtype=torch.float32)
        u_sample = torch.tensor(self.u[idx].values, dtype=torch.float32)
        x_sample = torch.tensor(self.x[idx].values, dtype=torch.float32)
        return c_sample, u_sample, x_sample

# Load and prepare data
ds = xr.open_dataset('../../graph_pde_solver/dataset/elasticity.nc')
# Reshape using xarray methods
c, u, x = ds['c'], ds['u'], ds['x']
train_size, val_size, test_size = 1024, 128, 256
train_ds = PoissonDataset(c[:train_size], u[:train_size], x[:train_size])
val_ds = PoissonDataset(c[train_size:train_size+val_size], u[train_size:train_size+val_size], x[train_size:train_size+val_size])
test_ds = PoissonDataset(c[-test_size:], u[-test_size:], x[-test_size:])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

# Initialize model
model = GINO(
    in_channels=1,
    out_channels=1,
    gno_coord_dim=2,
    in_gno_channel_mlp_hidden_layers=[64, 64, 64],
    out_gno_channel_mlp_hidden_layers=[64, 64],
    fno_in_channels=64,
    fno_n_modes=(16, 16),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

meshgrid = torch.meshgrid(torch.linspace(-0.5, 1.5, 64), torch.linspace(-0.5, 1.5, 64))
latent_queries = torch.stack(meshgrid, dim=-1).unsqueeze(0).to(device) # [1, 64, 64, 2]
# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for c_batch, u_batch, x_batch in loader:
        c_batch, u_batch, x_batch = c_batch.to(device), u_batch.to(device), x_batch.to(device)
        c_batch, u_batch, x_batch = c_batch.squeeze(1), u_batch.squeeze(1), x_batch.squeeze(1)
        optimizer.zero_grad()
        pred = model(x=c_batch, input_geom=x_batch[0:1], latent_queries=latent_queries, output_queries=x_batch[0]) # [n_batch, n_output_queries, 1]
        loss = criterion(pred, u_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for c_batch, u_batch, x_batch in loader:
            c_batch, u_batch, x_batch = c_batch.to(device), u_batch.to(device), x_batch.to(device)
            c_batch, u_batch, x_batch = c_batch.squeeze(1), u_batch.squeeze(1), x_batch.squeeze(1)
            
            pred = model(x=c_batch, input_geom=x_batch[0:1], latent_queries=latent_queries, output_queries=x_batch[0])
            loss = criterion(pred, u_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def errors_compute(model, loader, device):
    model.eval()
    l1_error, l2_error = 0.0, 0.0
    l1_norm, l2_norm = 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for c_batch, u_batch, x_batch in loader:
            c_batch, u_batch, x_batch = c_batch.to(device), u_batch.to(device), x_batch.to(device)
            c_batch, u_batch, x_batch = c_batch.squeeze(1), u_batch.squeeze(1), x_batch.squeeze(1)
            
            pred = model(x=c_batch, input_geom=x_batch[0:1], latent_queries=latent_queries, output_queries=x_batch[0])
            
            diff = pred - u_batch
            l1_error += torch.sum(torch.abs(diff)).item()
            l2_error += torch.sum(diff**2).item()
            
            l1_norm += torch.sum(torch.abs(u_batch)).item()
            l2_norm += torch.sum(u_batch**2).item()
            
            total_samples += u_batch.numel()

    l1_error /= total_samples
    l2_error = np.sqrt(l2_error / total_samples)
    l1_norm /= total_samples
    l2_norm = np.sqrt(l2_norm / total_samples)

    relative_l1 = l1_error / l1_norm
    relative_l2 = l2_error / l2_norm

    return {
        'L1': l1_error,
        'L2': l2_error,
        'Relative L1': relative_l1,
        'Relative L2': relative_l2
    }

# Training loop
num_epochs = 1000
best_val_loss = float('inf')

pbar = tqdm(total=num_epochs, desc="Epochs", colour = "blue")
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    pbar.update(1)
    pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'test_model/best_gino_elasticity_model.pth')
pbar.close()
# Test the model
model.load_state_dict(torch.load('test_model/best_gino_elasticity_model.pth'))
test_errors = errors_compute(model, test_loader, device)
print("Test Errors:")
for error_name, error_value in test_errors.items():
    print(f"{error_name}: {error_value:.6f}")

# 将结果保存到文件
with open('test_model/gino_elasticity_errors.txt', 'w') as f:
    for error_name, error_value in test_errors.items():
        f.write(f"{error_name}: {error_value:.6f}\n")

# Visualize predictions
model.eval()
with torch.no_grad():
    c_sample, u_sample, x_sample = next(iter(test_loader))
    c_sample, u_sample, x_sample = c_sample.to(device), u_sample.to(device), x_sample.to(device)
    c_sample, u_sample, x_sample = c_sample.squeeze(1), u_sample.squeeze(1), x_sample.squeeze(1)
    
    pred = model(x=c_sample, input_geom=x_sample[0:1], latent_queries=latent_queries, output_queries=x_sample[0])

    # Plot the first sample in the batch
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = ax1.tricontourf(x_sample[0, :, 0].cpu(), x_sample[0, :, 1].cpu(), c_sample[0, :, 0].cpu(), cmap='plasma')
    ax1.set_title("Input (c)")
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.tricontourf(x_sample[0, :, 0].cpu(), x_sample[0, :, 1].cpu(), u_sample[0, :, 0].cpu(), cmap='plasma')
    ax2.set_title("Ground Truth (u)")
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    im3 = ax3.tricontourf(x_sample[0, :, 0].cpu(), x_sample[0, :, 1].cpu(), pred[0, :, 0].cpu(), cmap='plasma')
    ax3.set_title("Prediction")
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('test_model/gino_elasticity_prediction.png')
    plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(train_losses, label='Train Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_yscale('log')
ax[0].legend()
ax[0].set_title('Training and Validation Loss')
ax[1].plot(val_losses, label='Validation Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_title('Validation Loss')
plt.savefig('test_model/gino_elasticity_loss.png')
plt.close()



print(f"Total number of model parameters: {count_model_params(model)}")