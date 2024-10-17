import torch 
import torch.nn as nn
from typing import Optional

from .p2r2p import Physical2Regional2Physical
from .cmpt.attn import Transformer, TransformerConfig
from .cmpt.deepgnn import DeepGraphNNConfig
from ..utils.dataclass import shallow_asdict
from ..graph import RegionInteractionGraph, Graph

class RANO(Physical2Regional2Physical):
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 rigraph:RegionInteractionGraph,
                 variable_mesh:bool = False,
                 drop_edge:float = 0.0,
                 patch_size: int = 8,
                 gnnconfig:DeepGraphNNConfig = DeepGraphNNConfig(),
                 attnconfig:TransformerConfig = TransformerConfig(),
                 regional_points:tuple = (64,64)):
        nn.Module.__init__(self)
        self.drop_edge = drop_edge
        self.input_size = input_size 
        self.output_size = output_size
        self.node_latent_size = gnnconfig.node_latent_size
        self.patch_size = patch_size
        self.H = regional_points[0]
        self.W = regional_points[1]

        self.encoder   = self.init_encoder(input_size, rigraph, gnnconfig)
        self.processor = self.init_processor(self.node_latent_size,  attnconfig)
        self.decoder   = self.init_decoder(output_size, rigraph, variable_mesh, gnnconfig)

    def init_processor(self, node_latent_size, config):
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * self.node_latent_size,
                                      self.patch_size * self.patch_size * self.node_latent_size)
        self.register_buffer('positional_embeddings', self.get_positional_embeddings())
       
        return Transformer(
            input_size = node_latent_size * self.patch_size * self.patch_size,
            output_size = node_latent_size * self.patch_size * self.patch_size,
            config = config
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
        n_regional_nodes = rndata.shape[-2]
        C = rndata.shape[2]  
        H = self.H
        W = self.W
        assert n_regional_nodes == H * W, f"n_regional_nodes({n_regional_nodes}) is not equal to H({H}) * W({W})"
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"H({H}) and W({W}) must be divisible by P({P})"
        num_patches_H = H // P
        num_patches_W = W // P
        num_patches = num_patches_H * num_patches_W
        # reshape to patches
        rndata = rndata.view(batch_size, H, W, C)
        rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()  # (batch_size, num_patches_H, num_patches_W, P, P, C)
        rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)  # (batch_size, num_patches, P*P*C)
        
        # ViT
        rndata = self.patch_linear(rndata)
        pos_emb = self.positional_embeddings.unsqueeze(0)  # Shape: (1, num_patches, embed_dim)
        rndata = rndata + pos_emb
        rndata = self.processor(rndata, condition = condition)

        # reshape back to the original shape
        rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, H * W, C)

        return rndata

    def get_positional_embeddings(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_H = self.H // self.patch_size
        num_patches_W = self.W // self.patch_size
        positions = torch.stack(torch.meshgrid(
            torch.arange(num_patches_H, dtype=torch.float32),
            torch.arange(num_patches_W, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)  # Shape: (num_patches, 2)
        pos_emb = self.compute_rope_embeddings(positions, self.patch_size * self.patch_size * self.node_latent_size)
        return pos_emb
    
    def compute_rope_embeddings(self, positions, embed_dim):
        """
        Compute Rope embeddings for the given positions.
        """
        # RoPE implementation
        positions = positions
        num_pos_dims = positions.size(1)  
        dim_touse = embed_dim // (2 * num_pos_dims)  
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        
        # Compute sinusoidal input: positions [num_patches, 2], inv_freq [dim_touse]
        # Resulting in [num_patches, 2, dim_touse]
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        
        # Apply sin and cos, resulting in [num_patches, 2, 2*dim_touse]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        
        # Flatten to get [num_patches, 2 * 2 * dim_touse] = [num_patches, embed_dim]
        pos_emb = pos_emb.view(positions.size(0), -1)
        
        return pos_emb  # Shape: (num_patches, embed_dim)

