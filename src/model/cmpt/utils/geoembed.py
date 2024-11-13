import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum, scatter_max

class GeometricEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, method='statistical', **kwargs):
        super(GeometricEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method.lower()
        self.kwargs = kwargs 

        if self.method == 'statistical':
            self.mlp = nn.Sequential(
                nn.Linear(self._get_stat_feature_dim(), 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.ReLU()
            )
            self.register_buffer("geo_features_normalized_cache", None)
        elif self.method == 'pointnet':
            self.pointnet_mlp = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(64, output_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def forward(self, input_geom, latent_queries, spatial_nbrs):
        if self.method == 'statistical':
            if self.geo_features_normalized_cache is None:
                geo_features_normalized = self._compute_statistical_features(input_geom, latent_queries, spatial_nbrs)
                self.geo_features_normalized_cache = geo_features_normalized
            else:
                geo_features_normalized = self.geo_features_normalized_cache
            return self.mlp(geo_features_normalized)
        elif self.method == 'pointnet':
            return self._compute_pointnet_features(input_geom, latent_queries, spatial_nbrs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_stat_feature_dim(self):
        num_features = 3 + 2 * self.input_dim
        return num_features
    
    def _compute_statistical_features(self, input_geom, latent_queries, spatial_nbrs):
            """
            Parameters:
                input_geom (torch.FloatTensor): The input geometry, shape: [num_nodes, num_dims]
                latent_queries (torch.FloatTensor): The latent queries, shape: [num_nodes, num_dims]
                spatial_nbrs (dict): {"neighbors_index": torch.LongTensor, "neighbors_row_splits": torch.LongTensor}             
                        neighbors_index: torch.Tensor with dtype=torch.int64
                            Index of each neighbor in data for every point
                            in queries. Neighbors are ordered in the same orderings
                            as the points in queries. Open3d and torch_cluster
                            implementations can differ by a permutation of the 
                            neighbors for every point.
                        neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                            The value at index j is the sum of the number of
                            neighbors up to query point j-1. First element is 0
                            and last element is the total number of neighbors.
            
            Returns:
                geo_features_normalized (torch.FloatTensor): The normalized geometric features, shape: [num_query_nodes, num_dims]
            """
            num_queries = latent_queries.shape[0]
            num_dims = latent_queries.shape[1]

            device = latent_queries.device

            neighbors_index = spatial_nbrs["neighbors_index"] # Shape: [num_total_neighbors]
            neighbors_row_splits = spatial_nbrs['neighbors_row_splits']  # Shape: [num_queries + 1]

            num_neighbors_per_query = neighbors_row_splits[1:] - neighbors_row_splits[:-1]  # Shape: [num_queries]
            query_indices_per_neighbor = torch.repeat_interleave(torch.arange(num_queries, device=device), num_neighbors_per_query)
            # Shape: [num_total_neighbors]

            nbr_coords = input_geom[neighbors_index]  # Shape: [num_total_neighbors, num_dims]
            query_coords_per_neighbor = latent_queries[query_indices_per_neighbor]  # Shape: [num_total_neighbors, num_dims]

            distances = torch.norm(nbr_coords - query_coords_per_neighbor, dim=1)  # Shape: [num_total_neighbors]
            N_i = num_neighbors_per_query.float()
            has_neighbors = N_i > 0 # Shape: [num_queries,]
            D_avg = scatter_mean(distances, query_indices_per_neighbor, dim=0, dim_size=num_queries)  # Shape: [num_queries,]

            distances_squared = distances ** 2
            E_X2 = scatter_mean(distances_squared, query_indices_per_neighbor, dim=0, dim_size=num_queries)  # Shape: [num_queries,]
            E_X = D_avg
            E_X_squared = E_X ** 2 # Shape: [num_queries,]

            D_var = E_X2 - E_X_squared  # Shape: [num_queries,]
            D_var = torch.clamp(D_var, min=0.0)  # Shape: [num_queries,]

            nbr_centroid = scatter_mean(nbr_coords, query_indices_per_neighbor, dim=0, dim_size=num_queries)  # Shape: [num_queries, num_dims]
            Delta = nbr_centroid - latent_queries 
            nbr_coords_centered = nbr_coords - nbr_centroid[query_indices_per_neighbor]  # Shape: [num_total_neighbors, num_dims]

            cov_components = nbr_coords_centered.unsqueeze(2) * nbr_coords_centered.unsqueeze(1)  # Shape: [num_total_neighbors, num_dims, num_dims]
            cov_sum = scatter_sum(cov_components, query_indices_per_neighbor, dim=0, dim_size=num_queries)  # Shape: [num_queries, num_dims, num_dims]

            N_i_clamped = N_i.clone()
            N_i_clamped[N_i_clamped == 0] = 1.0  # Prevent division by zero
            cov_matrix = cov_sum / N_i_clamped.view(-1, 1, 1)  # Shape: [num_queries, num_dims, num_dims]

            # Initialize PCA features tensor
            PCA_features = torch.zeros(num_queries, num_dims, device=device)

            # For queries with neighbors, compute eigenvalues
            if has_neighbors.any():
                # Extract covariance matrices for queries with neighbors
                cov_matrix_valid = cov_matrix[has_neighbors]  # Shape: [num_valid_queries, num_dims, num_dims]

                # Compute eigenvalues (since covariance matrices are symmetric)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix_valid)  # Shape: [num_valid_queries, num_dims]

                # Flip eigenvalues to descending order
                eigenvalues = eigenvalues.flip(dims=[1])

                # Assign eigenvalues to PCA_features
                PCA_features[has_neighbors] = eigenvalues

            # Stack all features
            N_i_tensor = N_i.unsqueeze(1)  # Shape: [num_queries, 1]
            D_avg_tensor = D_avg.unsqueeze(1)  # Shape: [num_queries, 1]
            D_var_tensor = D_var.unsqueeze(1)  # Shape: [num_queries, 1]

            # Combine features
            geo_features = torch.cat([N_i_tensor, D_avg_tensor, D_var_tensor, Delta, PCA_features], dim=1)
            # Shape: [num_queries, num_features]

            # Optionally, set features of queries with zero neighbors to zero
            geo_features[~has_neighbors] = 0.0

            # Feature normalization (Standardization)
            # Compute mean and std, avoiding zero std
            feature_mean = geo_features.mean(dim=0, keepdim=True)
            feature_std = geo_features.std(dim=0, keepdim=True)
            feature_std[feature_std < 1e-6] = 1.0  # Prevent division by near-zero std

            # Normalize features
            geo_features_normalized = (geo_features - feature_mean) / feature_std

            return geo_features_normalized

    def _compute_pointnet_features(self, input_geom, latent_queries, spatial_nbrs):
        """
        Use pointnet to compute geometric features for each query point.

        Parameters:
            input_geom (torch.FloatTensor): The input geometry, shape: [num_nodes, num_dims]
            latent_queries (torch.FloatTensor): The latent queries, shape: [num_nodes, num_dims]
            spatial_nbrs (dict): {"neighbors_index": torch.LongTensor, "neighbors_row_splits": torch.LongTensor}
                    neighbors_index: torch.Tensor with dtype=torch.int64
                        Index of each neighbor in data for every point
                        in queries. Neighbors are ordered in the same orderings
                        as the points in queries. Open3d and torch_cluster
                        implementations can differ by a permutation of the
                        neighbors for every point.
                    neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                        The value at index j is the sum of the number of
                        neighbors up to query point j-1. First element is 0
                        and last element is the total number of neighbors.
        Returns:
            geo_features (torch.FloatTensor): The geometric features, shape: [num_query_nodes, num_dims]
        """

        num_queries = latent_queries.shape[0]
        device = latent_queries.device

        neighbors_index = spatial_nbrs["neighbors_index"]  # [num_total_neighbors]
        neighbors_row_splits = spatial_nbrs['neighbors_row_splits']  # [num_queries + 1]

        num_neighbors_per_query = neighbors_row_splits[1:] - neighbors_row_splits[:-1]  # [num_queries]
        has_neighbors = num_neighbors_per_query > 0  # [num_queries]

        # 对于没有邻居的查询点，我们可以创建一个零向量或者一个可学习的参数
        geo_features = torch.zeros(num_queries, self.output_dim, device=device)

        # 仅对有邻居的查询点进行处理
        if has_neighbors.any():
            # 获取有邻居的查询点的索引
            valid_query_indices = torch.nonzero(has_neighbors).squeeze(1)
            # 获取有邻居的查询点对应的邻居起始和结束索引
            valid_starts = neighbors_row_splits[:-1][has_neighbors]
            valid_ends = neighbors_row_splits[1:][has_neighbors]

            # 为每个邻居点分配其所属的查询点索引
            query_indices_per_neighbor = torch.repeat_interleave(valid_query_indices, num_neighbors_per_query[has_neighbors])

            # 获取邻居点坐标
            nbr_coords = input_geom[neighbors_index]  # [num_total_neighbors, num_dims]
            # 获取对应的查询点坐标
            query_coords_per_neighbor = latent_queries[query_indices_per_neighbor]  # [num_total_valid_neighbors, num_dims]

            # 将邻居点相对于查询点进行中心化
            nbr_coords_centered = nbr_coords - query_coords_per_neighbor  # [num_total_valid_neighbors, num_dims]

            # 通过 MLP 逐点处理邻居点
            nbr_features = self.pointnet_mlp(nbr_coords_centered)  # [num_total_valid_neighbors, feature_dim]

            # 使用 scatter_max 在每个查询点的邻域内进行最大池化
            # 首先，我们需要为每个邻居点提供其对应的查询点索引
            # query_indices_per_neighbor 已经是我们需要的索引

            # 使用 scatter_max 进行最大池化
            max_features, _ = scatter_max(nbr_features, query_indices_per_neighbor, dim=0, dim_size=num_queries)

            # 对于有邻居的查询点，获取其对应的最大特征
            pointnet_features = max_features[valid_query_indices]  # [num_valid_queries, feature_dim]

            # 通过全连接层
            pointnet_features = self.fc(pointnet_features)  # [num_valid_queries, output_dim]

            # 将特征赋值给对应的查询点
            geo_features[valid_query_indices] = pointnet_features

        return geo_features