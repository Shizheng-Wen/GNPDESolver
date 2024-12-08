import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field

from .p2r2p import Physical2Regional2Physical
from .cmpt.attn import Transformer, TransformerConfig
from .cmpt.gno_triplane import IntegralTransform, GNOConfig, GNOEncoder, GNODecoder
from .cmpt.mlp import LinearChannelMLP
from .cmpt.utils.gno_utils import NeighborSearch
from ..graph import RegionInteractionGraph, Graph


class Triplane(Physical2Regional2Physical):
    """
    LANO: Graph Neural Operator + (U) Vision Transformer + Graph Neural Operator
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rigraph: RegionInteractionGraph,
                 variable_mesh: bool = False,
                 drop_edge: float = 0.0,
                 gno_config: GNOConfig = GNOConfig(),
                 attn_config: TransformerConfig = TransformerConfig(),
                 regional_points: tuple = (64, 64)):
        nn.Module.__init__(self)
        self.drop_edge = drop_edge
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = gno_config.lifting_channels 
        self.patch_size = attn_config.patch_size
        self.H = regional_points[0]
        self.W = regional_points[1]
        self.axis_concatenated = attn_config.axis_concatenated
        
        # Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, rigraph, gno_config)
        self.processor = self.init_processor(self.node_latent_size, attn_config)
        self.decoder = self.init_decoder(output_size, rigraph, variable_mesh, gno_config)
    
    def init_encoder(self, input_size, rigraph, gno_config):
        self.axisx_projection = LinearChannelMLP(layers=[input_size, self.node_latent_size])
        self.axisy_projection = LinearChannelMLP(layers=[input_size, self.node_latent_size])
        return GNOEncoder(
            in_channels = self.node_latent_size,
            out_channels = self.node_latent_size,
            gno_config = gno_config
        )
        
    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor

        self.positional_embedding_name = config.positional_embedding
        self.positions = self.get_patch_positions()

        if self.positional_embedding_name == 'absolute':
            num_x_patches = int(self.W)
            num_y_patches = int(self.H)
            pos_emb = self.compute_absolute_embeddings(self.positions, self.node_latent_size).reshape(num_x_patches,num_y_patches, -1)
            x_pos_emb = pos_emb[num_x_patches//2-1,:,:]
            y_pos_emb = pos_emb[:,num_y_patches//2-1,:]
            self.register_buffer('x_positional_embeddings', x_pos_emb)
            self.register_buffer('y_positional_embeddings', y_pos_emb)
        
        setattr(config.attn_config, 'H', self.H)
        setattr(config.attn_config, 'W', self.W)
        
        return Transformer(
            input_size=self.node_latent_size,
            output_size=self.node_latent_size,
            config=config
        )
    
    def init_decoder(self, output_size, rigraph, variable_mesh, gno_config):
        # Initialize the GNO decoder
        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(2 * self.node_latent_size, self.node_latent_size),
            nn.ReLU(),
            nn.Linear(self.node_latent_size, output_size)
        )
        return GNODecoder(
            in_channels=self.node_latent_size,
            out_channels=self.node_latent_size,
            gno_config=gno_config
        )

    def encode(self, graph: RegionInteractionGraph, pndata: torch.Tensor) -> torch.Tensor:
        # Apply GNO encoder
        x_projected_data = self.axisx_projection(pndata)
        y_projected_data = self.axisy_projection(pndata)

        input_geom = graph.physical_to_regional.src_ndata['pos']
        x_coords = input_geom[:, 0:1]
        y_coords = input_geom[:, 1:2]
        num_queries = 64
        queries = torch.linspace(-1, 1, steps=num_queries).unsqueeze(-1).to(pndata.device)  # [num_queries, 1]

        x_encoded = self.encoder(
            input_coords = x_coords,
            queries = queries,
            pndata = x_projected_data,
            axis_key="x"
        )

        y_encoded = self.encoder(
            input_coords = y_coords,
            queries = queries,
            pndata = y_projected_data,
            axis_key="y"
        )
        
        return x_encoded, y_encoded

    def process(self, graph: Graph,
                rndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        graph:Graph
            regional to regional graph, a homogeneous graph
        rndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        x_encoded = rndata[0]
        y_encoded = rndata[1]

        batch_size, seq_len_x, hidden_dim = x_encoded.shape
        _, seq_len_y, _ = y_encoded.shape

        if self.positional_embedding_name == 'absolute':
            x_encoded = x_encoded + self.x_positional_embeddings
            y_encoded = y_encoded + self.y_positional_embeddings
            relative_positions = None
        if self.axis_concatenated:
            combined = torch.cat([x_encoded, y_encoded], dim=1) # [batch_size, seq_len_x+seq_len_y,hidden_dim]
            combined_processed = self.processor(combined, condition=condition, relative_positions=relative_positions)
            x_processed = combined_processed[:, :seq_len_x, :]
            y_processed = combined_processed[:, seq_len_x:, :]
        else:
            x_processed = self.processor(x_encoded, condition=condition,relative_positions=relative_positions)
            y_processed = self.processor(y_encoded, condition=condition,relative_positions=relative_positions)

        return x_processed, y_processed

    def decode(self, graph: RegionInteractionGraph, rndata: torch.Tensor) -> torch.Tensor:
        # Apply GNO decoder
        x_processed, y_processed = rndata

        num_input_geom = 64
        input_geom = torch.linspace(-1, 1, steps=num_input_geom).unsqueeze(-1).to(x_processed.device)

        queries = graph.regional_to_physical.dst_ndata['pos']
        x_queries = queries[:,0:1]
        y_queries = queries[:,1:2]
        
        x_decoded = self.decoder(
            input_coords = input_geom,
            queries = x_queries,
            rndata = x_processed,
            axis_key = "x"
        )

        y_decoded = self.decoder(
            input_coords = input_geom,
            queries = y_queries,
            rndata = y_processed,
            axis_key = "y"
        )
        
        combined_features = torch.cat([x_decoded, y_decoded], dim = -1)
        output = self.reconstruction_mlp(combined_features)
    
        return output

    def forward(self,
                graphs: RegionInteractionGraph,
                pndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Forward pass for LANO model.

        Parameters
        ----------
        graphs: RegionInteractionGraph
            The graphs representing the interactions.
        pndata: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_physical_nodes, input_size]
        condition: Optional[float]
            The condition of the model

        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, n_physical_nodes, output_size]
        """
        # Encode: Map physical nodes to regional nodes using GNO Encoder
        rndata = self.encode(graphs, pndata)

        # Process: Apply Vision Transformer on the regional nodes
        rndata = self.process(graphs.regional_to_regional, rndata, condition)

        # Decode: Map regional nodes back to physical nodes using GNO Decoder
        output = self.decode(graphs, rndata)

        return output

    def get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_H = self.H
        num_patches_W = self.W
        positions = torch.stack(torch.meshgrid(
            torch.arange(num_patches_H, dtype=torch.float32),
            torch.arange(num_patches_W, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)

        return positions

    def compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute RoPE embeddings for the given positions.
        """
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb

