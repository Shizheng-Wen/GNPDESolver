import torch
import numpy as np

# 第一段代码
def radius_bipartite_graph(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    radii_b: torch.Tensor,
    periodic: bool = False,
    p: float = 2,
) -> torch.Tensor:
    assert points_a.ndim == 2 and points_b.ndim == 2, f"The points_a and points_b are expected to be 2D tensors, but got shapes {points_a.shape} and {points_b.shape}"
    assert points_b.shape[0] == radii_b.shape[0], f"The points_b and radii_b should have the same number of points, but got shapes {points_b.shape} and {radii_b.shape}"
    if periodic:
        residual = points_a[:, None, :] - points_b[None, :, :]
        residual = torch.where(residual >= 1., residual - 2., residual)
        residual = torch.where(residual < -1., residual + 2., residual)
        distances = torch.linalg.norm(residual, axis=-1, ord=p)  # [n_points_a, n_points_b]
    else:
        distances = torch.cdist(points_a, points_b, p=p)  # [n_points_a, n_points_b]

    bipartite_edges = torch.stack(torch.where(distances < radii_b[None, :]), 0)  # [2, n_edges]
    return bipartite_edges

# 第二段代码
def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    dists = torch.cdist(queries, data).to(queries.device)  # [n_queries, n_data]
    in_nbr = torch.where(dists <= radius, 1., 0.)  # [n_queries, n_data]
    nbr_indices = in_nbr.nonzero()  # [n_neighbors, 2], columns are [query_idx, data_idx]
    nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0)  # [n_queries]
    splits = torch.cat((torch.tensor([0]).to(queries.device), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices[:, 1].long()  # data indices
    nbr_dict['neighbors_row_splits'] = splits.long()
    nbr_dict['query_indices'] = nbr_indices[:, 0].long()  # query indices
    return nbr_dict

# 测试脚本
def test_neighbor_search():
    radius = 0.033 # 半径
    points_b = torch.from_numpy(np.load("elas_points.npz")["regional_points"])
    points_a = torch.from_numpy(np.load("elas_points.npz")["physical_points"])
    radii_b = torch.full((points_b.shape[0],), radius)
    # 使用第一段代码
    bipartite_edges = radius_bipartite_graph(points_a, points_b, radii_b, periodic=False, p=2)
    edges_from_first_code = list(zip(bipartite_edges[0].tolist(), bipartite_edges[1].tolist()))
    edges_from_first_code_set = set(edges_from_first_code)

    # 使用第二段代码
    nbr_dict = native_neighbor_search(data=points_a, queries=points_b, radius=radius)
    nbr_indices = nbr_dict['neighbors_index']
    query_indices = nbr_dict['query_indices']
    edges_from_second_code = list(zip(nbr_indices.tolist(), query_indices.tolist()))
    edges_from_second_code_set = set(edges_from_second_code)

    # 比较结果
    if edges_from_first_code_set == edges_from_second_code_set:
        print("两段代码产生的边完全一致。")
    else:
        print("两段代码产生的边存在差异。")
        edges_in_first_not_in_second = edges_from_first_code_set - edges_from_second_code_set
        edges_in_second_not_in_first = edges_from_second_code_set - edges_from_first_code_set
        print(f"第一段代码中有 {len(edges_in_first_not_in_second)} 条边不在第二段代码中。")
        print(f"第二段代码中有 {len(edges_in_second_not_in_first)} 条边不在第一段代码中。")

# 运行测试
test_neighbor_search()