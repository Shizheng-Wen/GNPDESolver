import numpy as np
import netCDF4 as nc
import os

# 列出要处理的数据文件
data_files = [
    "dataset/wave_dataset_0_1000.npz",
    "dataset/wave_dataset_1000_2000.npz",
    "dataset/wave_dataset_2000_3000.npz",
    "dataset/wave_dataset_3000_4000.npz",
    "dataset/wave_dataset_4000_5000.npz"
]

# 初始化坐标数组
x_initialized = False

# 创建新的 NetCDF 文件并准备维度和变量
output_file = "wave_c_sines_uv.nc"
with nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
    # 创建维度
    dst.createDimension('sample', None)
    dst.createDimension('time', 21)
    dst.createDimension('point', None)
    dst.createDimension('coord_dim', 2)
    dst.createDimension('channel', 1)
    dst.createDimension('sample_x', 1)
    dst.createDimension('time_x', 1)

    # 创建变量
    u_var = dst.createVariable('u', 'f4', ('sample', 'time', 'point', 'channel'))
    v_var = dst.createVariable('c', 'f4', ('sample', 'time', 'point', 'channel'))
    x_var = dst.createVariable('x', 'f4', ('sample_x', 'time_x', 'point', 'coord_dim'))

    # 可选：添加描述
    u_var.description = '选定时间步的位移场 u'
    v_var.description = '由 u 计算的速度场 v'
    x_var.description = '空间坐标 x 和 y'

    sample_index = 0  # 用于跟踪当前的样本索引

    # 遍历每个文件并处理数据
    for file_path in data_files:
        print(f"Processing {file_path}...")
        # 加载数据
        wave_data = np.load(file_path)
        wave_x = wave_data["x"].astype(np.float32)  # 转换为 float32
        wave_y = wave_data["y"].astype(np.float32)
        wave_u = wave_data["u"].astype(np.float32)  # 形状: (num_samples, num_time_steps, num_points)
        wave_dt = wave_data["dt"].item()  # 确保 dt 是标量

        # 获取维度信息
        num_samples, num_time_steps, num_points = wave_u.shape

        # 如果尚未初始化 point 维度的大小
        if dst.dimensions['point'].size is None:
            dst.dimensions['point'].size = num_points

        # 计算速度 v
        wave_v = np.zeros_like(wave_u, dtype=np.float32)
        # 对内部时间步使用中心差分
        wave_v[:, 1:-1, :] = (wave_u[:, 2:, :] - wave_u[:, :-2, :]) / (2 * wave_dt)
        # 初始速度 v_0 设为零
        wave_v[:, 0, :] = 0.0
        # 对最后一个时间步使用后向差分
        wave_v[:, -1, :] = (wave_u[:, -1, :] - wave_u[:, -2, :]) / wave_dt

        # 选择 21 个时间步
        selected_indices = np.linspace(0, num_time_steps - 1, 21).astype(int)
        wave_u_selected = wave_u[:, selected_indices, :]
        wave_v_selected = wave_v[:, selected_indices, :]

        # 扩展维度以匹配所需的形状
        wave_u_selected = np.expand_dims(wave_u_selected, axis=-1)  # 形状: (num_samples, 21, num_points, 1)
        wave_v_selected = np.expand_dims(wave_v_selected, axis=-1)

        # 一次性准备坐标数组
        if not x_initialized:
            # 将 x 和 y 合并为形状 (1, 1, num_points, 2) 的数组
            coords = np.stack((wave_x, wave_y), axis=-1)  # 形状: (num_points, 2)
            coords = coords[np.newaxis, np.newaxis, :, :]  # 形状: (1, 1, num_points, 2)
            x_var[0, 0, :, :] = coords[0, 0, :, :]
            x_initialized = True

        # 写入数据到 NetCDF 变量
        num_new_samples = num_samples
        u_var[sample_index:sample_index+num_new_samples, :, :, :] = wave_u_selected
        v_var[sample_index:sample_index+num_new_samples, :, :, :] = wave_v_selected

        # 更新样本索引
        sample_index += num_new_samples

        # 删除变量以释放内存
        del wave_data, wave_u, wave_v, wave_u_selected, wave_v_selected

        print(f"Finished processing {file_path}.")

    print(f"Data successfully saved to {output_file}")