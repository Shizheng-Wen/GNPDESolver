import xarray as xr
import numpy as np
import os


input_dir = "/cluster/work/math/camlab-data/poseidon-data"
output_dir = "/cluster/work/math/camlab-data/graphnpde"
filenames = ["NS-Sines.nc", "CE-CRP.nc", "CE-KH.nc", "CE-RPUI.nc"]

for filename in filenames:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    print(f"Processing：{filename}")

    ds = xr.open_dataset(input_path)

    if 'sample' in ds.dims:
        sample_dim = 'sample'
    elif 'member' in ds.dims:
        sample_dim = 'member'
    else:
        raise ValueError(f"未知的样本维度名，在文件 {filename} 中找不到 'sample' 或 'member' 维度。")
    print(f"样本维度名：{sample_dim}")

    if 'velocity' in ds.data_vars:
        data_var_name = 'velocity'
    elif 'data' in ds.data_vars:
        data_var_name = 'data'
    else:
        raise ValueError(f"未知的数据变量名，在文件 {filename} 中找不到 'velocity' 或 'data' 变量。")
    print(f"数据变量名：{data_var_name}")


    num_samples = ds.sizes[sample_dim]
    num_select = 3000
    random_MODE = False
    if random_MODE:
        selected_indices = np.random.choice(num_samples, size=num_select, replace=False)
    else:
        selected_indices = np.arange(num_select)
    output = ds[data_var_name][selected_indices].values
    output = output.reshape(*output.shape[0:3], -1).transpose(0, 1, 3, 2)
    del ds

    grid_size = 128
    x_coords = np.linspace(0., 1., grid_size)
    y_coords = np.linspace(0., 1., grid_size)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    coords = np.stack([X_flat, Y_flat], axis=-1)  # 形状为(16384, 2)
    coords_expanded = coords[np.newaxis, np.newaxis, :, :]  # (1, 1, 16384, 2)

    # 创建'x'变量
    x_da = xr.DataArray(coords_expanded, dims=("dim1", 'dim2', 'nodes', 'coord_dim'))
    u_da = xr.DataArray(output, dims =("sample", "time", "nodes", "channels"))


    ds = xr.Dataset(
        {
            "x": x_da,
            "u": u_da
        }
    )

    # 保存处理后的数据集
    ds.to_netcdf(output_path)

    print(f"已成功处理并保存文件：{output_path}\n")