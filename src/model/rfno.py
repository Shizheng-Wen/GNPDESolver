import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field

from .p2r2p import Physical2Regional2Physical
from .cmpt.fno import FNOBlocks, FNOConfig
from .cmpt.mlp import ChannelMLP
from .cmpt.deepgnn import DeepGraphNNConfig
from ..utils.dataclass import shallow_asdict
from ..graph import RegionInteractionGraph, Graph



class RFNO(Physical2Regional2Physical):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rigraph: RegionInteractionGraph,
                 variable_mesh: bool = False,
                 drop_edge: float = 0.0,
                 gnnconfig: DeepGraphNNConfig = DeepGraphNNConfig(),
                 fno_config: FNOConfig = FNOConfig(),
                 regional_points: tuple = (64, 64)):
        nn.Module.__init__(self)
        self.drop_edge = drop_edge
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = gnnconfig.node_latent_size
        self.H = regional_points[0]
        self.W = regional_points[1]

        self.encoder = self.init_encoder(input_size, rigraph, gnnconfig)
        self.processor = self.init_processor(self.node_latent_size, fno_config)
        self.decoder = self.init_decoder(output_size, rigraph, variable_mesh, gnnconfig)

    def init_processor(self, node_latent_size, fno_config):
        self.lifting = ChannelMLP(
            in_channels= self.node_latent_size,
            hidden_channels=fno_config.lifting_channels,
            out_channels=fno_config.hidden_channels,
            n_layers = 3
        )

        self.projection = ChannelMLP(
            in_channels = fno_config.hidden_channels,
            hidden_channels = fno_config.lifting_channels,
            out_channels = self.node_latent_size,
            n_layers = 3
        )

        kwargs = shallow_asdict(fno_config)
        kwargs.pop("lifting_channels")

        return FNOBlocks(
            in_channels = fno_config.hidden_channels,
            out_channels= fno_config.hidden_channels,
            **kwargs
            )

    def process(self, graph: Graph,
                rndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        graph: Graph
            Regional to regional graph, a homogeneous graph.
        rndata: Optional[torch.Tensor]
            Tensor of shape [batch_size, n_regional_nodes, node_latent_size].
        condition: Optional[float]
            The condition of the model.

        Returns
        -------
        torch.Tensor
            The regional node data of shape [batch_size, n_regional_nodes, node_latent_size].
        """
        batch_size, n_regional_nodes, C = rndata.shape
        H, W = self.H, self.W
        assert n_regional_nodes == H * W, \
            f"n_regional_nodes ({n_regional_nodes}) is not equal to H ({H}) * W ({W})"

        # Reshape rndata to (batch_size, C, H, W)
        rndata = rndata.view(batch_size, H, W, C).permute(0, 3, 1, 2).contiguous()

        rndata = self.lifting(rndata)

        # Apply FNO
        for idx in range(self.processor.n_layers):
            rndata = self.processor(rndata, idx)

        rndata = self.projection(rndata)
        # Reshape back to (batch_size, n_regional_nodes, C)
        rndata = rndata.permute(0, 2, 3, 1).contiguous().view(batch_size, n_regional_nodes, C)

        return rndata
