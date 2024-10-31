import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .p2r2p import Physical2Regional2Physical
from .cmpt.attn import Transformer, TransformerConfig
from .cmpt.deepgnn import DeepGraphNNConfig
from ..utils.dataclass import shallow_asdict
from ..graph import RegionInteractionGraph, Graph

class RANO(Physical2Regional2Physical):
    """
    RANO: Message Passing + (U) Vision Transformer + Message Passing
    """
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 rigraph:RegionInteractionGraph,
                 variable_mesh:bool = False,
                 drop_edge:float = 0.0,
                 gnnconfig:DeepGraphNNConfig = DeepGraphNNConfig(),
                 attnconfig:TransformerConfig = TransformerConfig(),
                 regional_points:tuple = (64,64)):
        nn.Module.__init__(self)
        self.drop_edge = drop_edge
        self.input_size = input_size 
        self.output_size = output_size
        self.node_latent_size = gnnconfig.node_latent_size
        self.patch_size = attnconfig.patch_size
        self.H = regional_points[0]
        self.W = regional_points[1]

        self.encoder   = self.init_encoder(input_size, rigraph, gnnconfig)
        self.processor = self.init_processor(self.node_latent_size,  attnconfig)
        self.decoder   = self.init_decoder(output_size, rigraph, variable_mesh, gnnconfig)

    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * self.node_latent_size,
                                      self.patch_size * self.patch_size * self.node_latent_size)
        

        self.positional_embedding_name = config.positional_embedding
        self.positions = self.get_patch_positions()
        if self.positional_embedding_name == 'absolute':
            pos_emb = self.compute_absolute_embeddings(self.positions, self.patch_size * self.patch_size * self.node_latent_size)
            self.register_buffer('positional_embeddings', pos_emb)
        
        setattr(config.attn_config, 'H', self.H)
        setattr(config.attn_config, 'W', self.W)
        
        return Transformer(
            input_size=self.node_latent_size * self.patch_size * self.patch_size,
            output_size=self.node_latent_size * self.patch_size * self.patch_size,
            config=config
        )

    def process(self, graph:Graph, 
                rndata:Optional[torch.Tensor] = None, 
                condition:Optional[float] = None
                )->torch.Tensor:
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
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        H, W = self.H, self.W
        assert n_regional_nodes == H * W, \
            f"n_regional_nodes ({n_regional_nodes}) is not equal to H ({H}) * W ({W})"
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"H({H}) and W({W}) must be divisible by P({P})"
        num_patches_H = H // P
        num_patches_W = W // P
        num_patches = num_patches_H * num_patches_W
        # Reshape to patches
        rndata = rndata.view(batch_size, H, W, C)
        rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)
        
        # Apply Vision Transformer
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)  # shape [num_patches, 2]

        if self.positional_embedding_name == 'absolute':
            pos_emb = self.compute_absolute_embeddings(pos, self.patch_size * self.patch_size * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None
    
        elif self.positional_embedding_name == 'rope':
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        # Reshape back to the original shape
        rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, H * W, C)

        return rndata

    def get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_H = self.H // self.patch_size
        num_patches_W = self.W // self.patch_size
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

