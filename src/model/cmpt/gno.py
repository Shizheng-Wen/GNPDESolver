import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.gno_utils import Activation, segment_csr, NeighborSearch
from .utils.geoembed import GeometricEmbedding
from .mlp import LinearChannelMLP, ChannelMLP
from ...graph import RegionInteractionGraph

from dataclasses import dataclass, replace, field
from typing import Union, Tuple, Optional

@dataclass
class GNOConfig:
    gno_coord_dim: int = 2
    projection_channels: int = 256
    in_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64, 64])
    out_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64])
    lifting_channels: int = 16
    gno_radius: float = 0.033
    gno_use_open3d: bool = False
    in_gno_transform_type: str = 'linear'
    out_gno_transform_type: str = 'linear'
    gno_use_torch_scatter: str = True
    node_embedding: bool = False
    use_attn: Optional[bool] = None 
    attention_type: str = 'cosine'
    use_geoembed: bool = False
    embedding_method: str = 'statistical'



############
# Integral Transform (GNO)
############

class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)
    
    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    channel_mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class
    channel_mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        LinearChannelMLP class. Only used if channel_mlp_layers is
        given and channel_mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_torch_scatter : bool, default 'True'
        Whether to use torch_scatter's implementation of 
        segment_csr or our native PyTorch version. torch_scatter 
        should be installed by default, but there are known versioning
        issues on some linux builds of CPU-only PyTorch. Try setting
        to False if you experience an error from torch_scatter.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_torch_scatter=True,
        use_attn=None,
        coord_dim=None,
        attention_type='cosine'
    ):
        super().__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter
        self.use_attn = use_attn
        self.attention_type = attention_type

        if self.transform_type not in ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"]:
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
        
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")
            self.coord_dim = coord_dim

            if self.attention_type == 'dot_product':
                attention_dim = 64 
                self.query_proj = nn.Linear(self.coord_dim, attention_dim)
                self.key_proj = nn.Linear(self.coord_dim, attention_dim)
                self.scaling_factor = 1.0 / (attention_dim ** 0.5)
            elif self.attention_type == 'cosine':
                pass
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")


    """"
    

    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def forward(self, y, neighbors, x=None, f_y=None, weights=None):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y
        
        rep_features = y[neighbors["neighbors_index"]]

        # batching only matters if f_y (latent embedding) values are provided
        batched = False
        # f_y has a batch dim IFF batched=True
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors["neighbors_index"]]

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )

        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        # attention usage
        if self.use_attn:
            query_coords = self_features[:, :self.coord_dim]
            key_coords = rep_features[:, :self.coord_dim]

            if self.attention_type == 'dot_product':
                query = self.query_proj(query_coords)  # [num_neighbors, attention_dim]
                key = self.key_proj(key_coords)        # [num_neighbors, attention_dim]

                attention_scores = torch.sum(query * key, dim=-1) * self.scaling_factor  # [num_neighbors]

            elif self.attention_type == 'cosine':
                query_norm = F.normalize(query_coords, p=2, dim=-1)
                key_norm = F.normalize(key_coords, p=2, dim=-1)
                attention_scores = torch.sum(query_norm * key_norm, dim=-1)  # [num_neighbors]

            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")

            splits = neighbors["neighbors_row_splits"]
            attention_weights = self.segment_softmax(attention_scores, splits)
        else:
            attention_weights = None


        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features = self.channel_mlp(agg_features)
    
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features
        
        if self.use_attn:
            rep_features = rep_features * attention_weights.unsqueeze(-1)

        if weights is not None:
            assert weights.ndim == 1, "Weights must be of dimension 1 in all cases"
            nbr_weights = weights[neighbors["neighbors_index"]]
            # repeat weights along batch dim if batched
            if batched:
                nbr_weights = nbr_weights.repeat(
                    [batch_size] + [1] * nbr_weights.ndim
                )
            rep_features = nbr_weights * rep_features
            reduction = "sum"
        else:
            reduction = "mean" if not self.use_attn else "sum"

        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(rep_features, splits, reduce=reduction, use_scatter=self.use_torch_scatter)

        return out_features

    def segment_softmax(self, attention_scores, splits):
        """
        apply soft_max on every regional node neighbors.

        Parameters：
        - attention_scores: [num_neighbors]，attention scores
        - splits: neighbors split information

        Return：
        - attention_weights: [num_neighbors]，normalized attention scores
        """
        max_values = segment_csr(
            attention_scores, splits, reduce='max', use_scatter=self.use_torch_scatter
        )
        max_values_expanded = max_values.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_scores = attention_scores - max_values_expanded
        exp_scores = torch.exp(attention_scores)
        sum_exp = segment_csr(
            exp_scores, splits, reduce='sum', use_scatter=self.use_torch_scatter
        )
        sum_exp_expanded = sum_exp.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_weights = exp_scores / sum_exp_expanded
        return attention_weights

############
# GNO Encoder TODO: only for the batch data which share the same graph structure.
############
class GNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.gno_radius = gno_config.gno_radius
        self.graph_cache = None 

        in_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = gno_config.gno_coord_dim 
        if gno_config.node_embedding:
            in_kernel_in_dim = gno_config.gno_coord_dim * 4 * 2 * 2 # 32
            coord_dim = gno_config.gno_coord_dim * 4 * 2
        if gno_config.in_gno_transform_type == "nonlinear" or gno_config.in_gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += in_channels
        
        gno_config.in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        gno_config.in_gno_channel_mlp_hidden_layers.append(out_channels)
        self.gno = IntegralTransform(
            channel_mlp_layers= gno_config.in_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.in_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            coord_dim=coord_dim,
            attention_type=gno_config.attention_type
        )
        
        self.lifting = ChannelMLP(
            in_channels = in_channels,
            out_channels = out_channels,
            n_layers= 1
            )
        
        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim= gno_config.gno_coord_dim,
                output_dim=out_channels,
                method=gno_config.embedding_method
            )
            self.recovery = ChannelMLP(
                in_channels = 2 * out_channels,
                out_channels = out_channels,
                n_layers= 1
            )

    def forward(self, graph: RegionInteractionGraph, pndata: torch.Tensor) -> torch.Tensor:
        batch_size = pndata.shape[0]
        if self.graph_cache is None:
            self.input_geom = graph.physical_to_regional.src_ndata['pos'].to(pndata.device)
            self.latent_queries = graph.physical_to_regional.dst_ndata['pos'].to(pndata.device)
            self.spatial_nbrs = self.nb_search(
                self.input_geom,
                self.latent_queries,
                self.gno_radius
            )
            self.graph_cache = True
        pndata = pndata.permute(0,2,1)
        pndata = self.lifting(pndata).permute(0, 2, 1)  

        encoded = self.gno(
            y=graph.physical_to_regional.get_ndata()[0],
            x=graph.physical_to_regional.get_ndata()[1][:,:-1],
            f_y=pndata,
            neighbors=self.spatial_nbrs
        ) # [batch_size, num_nodes, channels]

        if hasattr(self, 'geoembed'):
            geoembedding = self.geoembed(
                self.input_geom,
                self.latent_queries,
                self.spatial_nbrs
            )
            geoembedding = geoembedding[None, :, :]
            geoembedding = geoembedding.repeat([batch_size, 1, 1])

            encoded = torch.cat([encoded, geoembedding], dim=-1)
            encoded = encoded.permute(0, 2, 1)
            encoded = self.recovery(encoded).permute(0, 2, 1)

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
            
        out_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = gno_config.gno_coord_dim 
        if gno_config.node_embedding:
            out_kernel_in_dim = gno_config.gno_coord_dim * 4 * 2 * 2 # 32
            coord_dim = gno_config.gno_coord_dim * 4 * 2
        out_kernel_in_dim += out_channels if gno_config.out_gno_transform_type != 'linear' else 0
        gno_config.out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        gno_config.out_gno_channel_mlp_hidden_layers.append(in_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers= gno_config.out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            coord_dim=coord_dim,
            attention_type=gno_config.attention_type
        )
        
        self.projection = ChannelMLP(
            in_channels = in_channels,
            out_channels = out_channels,
            hidden_channels = gno_config.projection_channels,
            n_layers=2,
            n_dim=1,
        )

        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim= gno_config.gno_coord_dim,
                output_dim=in_channels,
                method=gno_config.embedding_method
            )
            self.recovery = ChannelMLP(
                in_channels = 2 * in_channels,
                out_channels = in_channels,
                n_layers= 1
            )

    def forward(self, graph: RegionInteractionGraph, rndata: torch.Tensor) -> torch.Tensor:
        batch_size = rndata.shape[0]

        if self.graph_cache is None:
            self.input_geom = graph.regional_to_physical.src_ndata['pos'].to(rndata.device)
            self.latent_queries = graph.regional_to_physical.dst_ndata['pos'].to(rndata.device) 
            self.spatial_nbrs = self.nb_search(
                self.input_geom,
                self.latent_queries,
                self.gno_radius
            )
            self.graph_cache = True

        decoded = self.gno(
            y=graph.regional_to_physical.get_ndata()[0],
            x=graph.regional_to_physical.get_ndata()[1][:,:-1],
            f_y=rndata,
            neighbors=self.spatial_nbrs
        )

        if hasattr(self, 'geoembed'):
            geoembedding = self.geoembed(
                self.input_geom,
                self.latent_queries,
                self.spatial_nbrs
            )
            geoembedding = geoembedding[None, :, :]
            geoembedding = geoembedding.repeat([batch_size, 1, 1])

            decoded = torch.cat([decoded, geoembedding], dim=-1)
            decoded = decoded.permute(0, 2, 1)
            decoded = self.recovery(decoded).permute(0, 2, 1)

        decoded = decoded.permute(0,2,1)
        decoded = self.projection(decoded).permute(0, 2, 1)  


        return decoded