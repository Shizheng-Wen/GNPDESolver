import torch
import time 

def create_all_to_all_pairs(u_data, x_data, c_data):
    num_samples, num_timesteps, num_nodes, num_vars = u_data.shape
    device = u_data.device  # 确保张量在同一设备上

    # 创建所有可能的 t_in 和 t_out 组合，其中 t_out > t_in
    t_indices = torch.arange(num_timesteps, device=device)
    t_in_indices, t_out_indices = torch.meshgrid(t_indices, t_indices, indexing='ij')
    # 展平成一维
    t_in_indices = t_in_indices.flatten()  # 形状: [num_timesteps^2]
    t_out_indices = t_out_indices.flatten()

    # 只保留 t_out > t_in 的配对
    # max_time_diff = 5
    # time_diffs = t_out_indices - t_in_indices
    # mask = (t_out_indices > t_in_indices) & (time_diffs <= max_time_diff)
    mask = t_out_indices > t_in_indices
    t_in_indices = t_in_indices[mask]  # 形状: [num_pairs]
    t_out_indices = t_out_indices[mask]

    num_pairs = t_in_indices.shape[0]

    # 扩展样本索引
    sample_indices = torch.arange(num_samples, device=device).unsqueeze(1).repeat(1, num_pairs).flatten()  # 形状: [num_samples * num_pairs]

    # 为所有样本重复 t_in_indices 和 t_out_indices
    t_in_indices_expanded = t_in_indices.unsqueeze(0).repeat(num_samples, 1).flatten()  # 形状: [num_samples * num_pairs]
    t_out_indices_expanded = t_out_indices.unsqueeze(0).repeat(num_samples, 1).flatten()

    # 获取 u_in 和 u_out
    u_in = u_data[sample_indices, t_in_indices_expanded, :, :]  # 形状: [num_samples * num_pairs, num_nodes, num_vars]
    u_out = u_data[sample_indices, t_out_indices_expanded, :, :]  # 相同形状

    # 获取 x_in 和 x_out
    x_in = x_data[sample_indices, t_in_indices_expanded, :, :]  # 形状: [num_samples * num_pairs, num_nodes, num_dims]
    x_out = x_data[sample_indices, t_out_indices_expanded, :, :]  # 相同形状

    # 计算 lead_times 和 time_diffs
    lead_times = t_out_indices_expanded.unsqueeze(1).type(u_data.dtype)  # 形状: [num_samples * num_pairs, 1]
    time_diffs = (t_out_indices_expanded - t_in_indices_expanded).unsqueeze(1).type(u_data.dtype)  # 相同形状

    if c_data is not None:
        c_in = c_data[sample_indices, t_in_indices_expanded, :, :]  # 形状: [num_samples * num_pairs, num_nodes, num_c_vars]
    else:
        c_in = None

    return u_in, u_out, x_in, x_out, lead_times, time_diffs, c_in

# 示例数据
num_samples = 2000
num_timesteps = 21
num_nodes = 5
num_vars = 3
num_dims = 2

u_data = torch.randn(num_samples, num_timesteps, num_nodes, num_vars)
x_data = torch.randn(num_samples, num_timesteps, num_nodes, num_dims)
c_data = None  # 或者使用实际的 c_data

# 调用函数
start_time = time.time()
u_in, u_out, x_in, x_out, lead_times, time_diffs, c_in = create_all_to_all_pairs(u_data, x_data, c_data)

print("cost of time:", time.time() - start_time)
num_pairs_per_sample = num_timesteps * (num_timesteps - 1) // 2
num_pairs = num_pairs_per_sample * num_samples

print("u_in shape:", u_in.shape)
print("u_out shape:", u_out.shape)
print("x_in shape:", x_in.shape)
print("x_out shape:", x_out.shape)
print("lead_times shape:", lead_times.shape)
print("time_diffs shape:", time_diffs.shape)
print("num_pairs:", num_pairs)
breakpoint()
