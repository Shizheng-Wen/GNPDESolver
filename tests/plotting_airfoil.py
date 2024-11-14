import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 进度条库，可选
import xarray as xr

# 假设 c_array, u_array, 和 x_train 已经作为 NumPy 数组加载到内存中
# 如果数据存储在文件中，可以使用 np.load() 来加载
# 例如：
# c_array = np.load('c_array.npy')  # 形状: (4096, 1, 16384, 1)
# u_array = np.load('u_array.npy')  # 形状: (4096, 1, 16384, 1)
# x_train = np.load('x_train.npy')  # 形状: (1, 1, 16384, 2)
base_path = "/cluster/work/math/camlab-data/rigno-unstructured/"
dataset_name = "airfoil_grid"
dataset_path = os.path.join(base_path, f"{dataset_name}.nc")

with xr.open_dataset(dataset_path) as ds:
    u_array = ds["u"].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]

    c_array = ds["c"].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]

    domain_x = [[-0.75, -0.75], [1.75, 1.75]] #([xmin, ymin], [xmax, ymax])
    nx, ny = u_array.shape[-2], u_array.shape[-1]
    x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
    y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
    xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
    x_grid = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
    x_grid = x_grid.reshape(-1, 2)  # [num_nodes, num_dims]
    x_grid = x_grid[None, None, ...]  # Add sample and time dimensions
    x_train = x_grid 
    c_array = c_array.reshape(c_array.shape[0],-1)[:,None,:,None] # [num_samples, 1, num_nodes, 1]
    u_array = u_array.reshape(u_array.shape[0],-1)[:,None,:,None] # [num_samples, 1, num_nodes, 1]
        
samples = 500

c_array = c_array[:samples]
u_array = u_array[:samples]
x_train = x_train[:samples]


# 参数设置
num_samples = c_array.shape[0]  # 4096
output_dir = '../slurmjob/dataset_plots'     # 输出目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 确定网格大小
num_points = c_array.shape[2]  # 16384
grid_size = int(np.sqrt(num_points))
if grid_size ** 2 != num_points:
    raise ValueError(f"点的数量 {num_points} 不是一个完全平方数，无法重塑为正方形网格。")

# 提取x和y坐标
# x_train 的形状为 (1, 1, 16384, 2)
# 先重塑为 (16384, 2)
coordinates = x_train.reshape(-1, 2)
x_coords = coordinates[:, 0]
y_coords = coordinates[:, 1]

# 可选：确定 imshow 的范围
x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()
extent = [x_min, x_max, y_min, y_max]

# 遍历每个样本
for i in tqdm(range(num_samples), desc="绘制样本"):
    # 提取第i个样本的c和u数据
    c_sample = c_array[i, 0, :, 0]  # 形状: (16384,)
    u_sample = u_array[i, 0, :, 0]  # 形状: (16384,)
    
    # 重塑为 (128, 128)
    c_image = c_sample.reshape(grid_size, grid_size)
    u_image = u_sample.reshape(grid_size, grid_size)
    
    # 创建包含两个子图的图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制 c_array
    im0 = axes[0].imshow(c_image, extent=extent, origin='lower', aspect='auto')
    axes[0].set_title(f'样本 {i} - c_array')
    plt.colorbar(im0, ax=axes[0])
    
    # 绘制 u_array
    im1 = axes[1].imshow(u_image, extent=extent, origin='lower', aspect='auto')
    axes[1].set_title(f'样本 {i} - u_array')
    plt.colorbar(im1, ax=axes[1])
    
    # 设置轴标签
    axes[0].set_xlabel('X 坐标')
    axes[0].set_ylabel('Y 坐标')
    axes[1].set_xlabel('X 坐标')
    axes[1].set_ylabel('Y 坐标')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    filename = os.path.join(output_dir, f'sample_{i}.png')
    plt.savefig(filename)
    
    # 关闭图形以释放内存
    plt.close(fig)

print(f"所有样本已保存到目录：{output_dir}")