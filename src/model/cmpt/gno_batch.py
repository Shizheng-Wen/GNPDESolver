import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import numpy as np


from .mlp import LinearChannelMLP, ChannelMLP
from ...utils.scale import rescale
from ...utils.sample import subsample

from .utils.gno_utils import Activation, segment_csr, NeighborSearch
from .utils.geoembed import GeometricEmbedding

from ...graph import RegionInteractionGraph
from ...graph.support import minimal_support

from .gno import IntegralTransform


"""
This code is beta version, the final version will be integrated into the gno.py

I have droped the NeighborSearch_batch and IntegralTransformBatch. Because it is not a big 
for loop, we can use loop to reduce the peak of memory usage.
"""
SUBTOKEN = False
N_TOKEN = 1000
##########
# GNOEncoder
##########
class GNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.gno_radius = gno_config.gno_radius
        self.graph_cache = None 
        self.use_graph_cache = gno_config.use_graph_cache
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights

        in_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = gno_config.gno_coord_dim 
        if gno_config.node_embedding:
            in_kernel_in_dim = gno_config.gno_coord_dim * 4 * 2 * 2  # 32
            coord_dim = gno_config.gno_coord_dim * 4 * 2
        if gno_config.in_gno_transform_type == "nonlinear" or gno_config.in_gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += in_channels

        in_gno_channel_mlp_hidden_layers = gno_config.in_gno_channel_mlp_hidden_layers.copy()
        in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        in_gno_channel_mlp_hidden_layers.append(out_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.in_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            coord_dim=coord_dim,
            attention_type=gno_config.attention_type
        )

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=1
        )

        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=gno_config.gno_coord_dim,
                output_dim=out_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                n_layers=1
            )
        
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.coord_dim = coord_dim
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(self, graph: RegionInteractionGraph, pndata: torch.Tensor, x_coord: torch.Tensor):
        """
        graph: RegionInteractionGraph
        pndata: [n_batch, n, n_channels]
        x_coord: [n_batch, n, d]
        """
        x = rescale(x_coord)
        device = pndata.device
        latent_queries = graph.physical_to_regional.dst_ndata['pos'].to(device)
        pndata = pndata.permute(0,2,1)
        pndata = self.lifting(pndata).permute(0,2,1)

        n_batch, n, d = x_coord.shape
        m = latent_queries.shape[1]

        encoded = []
        for b in range(n_batch):
            x_b = x[b] # Shape: [n, d]
            pndata_b = pndata[b] # Shape: [n, n_channels]
            if SUBTOKEN:
                latent_queries = subsample(x_b, n = N_TOKEN)
            encoded_scales = []
            for scale in self.scales:
                scaled_radius = self.gno_radius * scale
                spatial_nbrs = self.nb_search(x_b, latent_queries, scaled_radius)
                encoded_unpatched = self.gno(
                    y = x_b,
                    x = latent_queries,
                    f_y = pndata_b,
                    neighbors = spatial_nbrs
                )

                if hasattr(self, 'geoembed'):
                    geoembedding = self.geoembed(
                        x_b,
                        latent_queries,
                        spatial_nbrs
                    ) # Shape: [n, d]

                    encoded_unpatched = torch.cat([encoded_unpatched, geoembedding], dim=-1)
                    encoded_unpatched = self.recovery(encoded_unpatched)
                encoded_scales.append(encoded_unpatched)
            
            if len(encoded_scales) == 1:
                encoded_data = encoded_scales[0]
            else:
                if self.use_scale_weights:
                    raise NotImplementedError
                else:
                    encoded_data = torch.stack(encoded_scales, 0).sum(dim=0)
            
            encoded.append(encoded_data)
        
        encoded = torch.stack(encoded, 0) # Shape: [n_batch, m, n_channels]
        return encoded


############
# GNO Decoder
############
class GNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.gno_radius = gno_config.gno_radius
        self.graph_cache = None
        self.use_graph_cache = gno_config.use_graph_cache
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights

        out_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = gno_config.gno_coord_dim 
        if gno_config.node_embedding:
            out_kernel_in_dim = gno_config.gno_coord_dim * 4 * 2 * 2  # 32
            coord_dim = gno_config.gno_coord_dim * 4 * 2
        if gno_config.out_gno_transform_type == "nonlinear" or gno_config.out_gno_transform_type == "nonlinear_kernelonly":
            out_kernel_in_dim += out_channels

        out_gno_channel_mlp_hidden_layers = gno_config.out_gno_channel_mlp_hidden_layers.copy()
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(in_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            coord_dim=coord_dim,
            attention_type=gno_config.attention_type
        )

        self.projection = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=gno_config.projection_channels,
            n_layers=2,
            n_dim=1,
        )

        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=gno_config.gno_coord_dim,
                output_dim=in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1
            )

        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.coord_dim = coord_dim
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)  

    def forward(self, graph: RegionInteractionGraph, rndata: torch.Tensor, x_coord: torch.Tensor):
        """
        graph: RegionInteractionGraph
        rndata: [n_batch, n, n_channels]
        x_coord: [n_batch, m, d]
        """
        device = rndata.device

        x = graph.regional_to_physical.src_ndata['pos'].to(device)
        latent_queries = rescale(x_coord)
        

        n_batch, n, d = latent_queries.shape

        decoded = []
        for b in range(n_batch):
            latent_queries_b = latent_queries[b] # Shape: [m, d]
            if SUBTOKEN:
                x = subsample(latent_queries_b, n = N_TOKEN)
            rndata_b = rndata[b] # Shape: [n, n_channels]
            decoded_scales = []
            for scale in self.scales:
                scaled_radius = self.gno_radius * scale
                spatial_nbrs = self.nb_search(x, latent_queries_b, scaled_radius)
                decoded_unpatched = self.gno(
                    y = x,
                    x = latent_queries_b,
                    f_y = rndata_b,
                    neighbors = spatial_nbrs
                )

                if hasattr(self, 'geoembed'):
                    geoembedding = self.geoembed(
                        x,
                        latent_queries_b,
                        spatial_nbrs
                    )

                    decoded_unpatched = torch.cat([decoded_unpatched, geoembedding], dim=-1)
                    decoded_unpatched = self.recovery(decoded_unpatched)
                decoded_scales.append(decoded_unpatched)
            
            if len(decoded_scales) == 1:
                decoded_data = decoded_scales[0]
            else:
                if self.use_scale_weights:
                    raise NotImplementedError
                else:
                    decoded_data = torch.stack(decoded_scales, 0).sum(dim=0)
            
            decoded.append(decoded_data)
        decoded = torch.stack(decoded, 0) # Shape: [n_batch, m, n_channels]
        decoded = decoded.permute(0,2,1)
        decoded = self.projection(decoded).permute(0, 2, 1)
        return decoded
