import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field
import math

from .p2r2p import Physical2Regional2Physical
from .cmpt.attn import Transformer, TransformerConfig
from .cmpt.gno import IntegralTransform, GNOConfig, GNOEncoder, GNODecoder
from .cmpt.utils.gno_utils import NeighborSearch
from .cmpt.scot import ScOT, SCOTConfig, ScOTConfig, ScOTOutput
from ..graph import RegionInteractionGraph, Graph
from ..utils.dataclass import shallow_asdict


class LSCOTProcessor(ScOT):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        embedding_output: Optional[torch.FloatTensor] = None,
        input_dimensions: Optional[Tuple[int, int]] = None,
        time: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ScOTOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if embedding_output is None or input_dimensions is None:
            raise ValueError("embedding_output and input_dimensions cannot be None")

        head_mask = self.get_head_mask(
            head_mask, self.num_layers_encoder + self.num_layers_decoder
        )

        if isinstance(head_mask, list):
            head_mask_encoder = head_mask[: self.num_layers_encoder]
            head_mask_decoder = head_mask[self.num_layers_encoder :]
        else:
            head_mask_encoder, head_mask_decoder = head_mask.split(
                [self.num_layers_encoder, self.num_layers_decoder]
            )

        # encoder part
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            time,
            head_mask=head_mask_encoder,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )

        if return_dict:
            skip_states = list(encoder_outputs.hidden_states[1:])
        else:
            skip_states = list(encoder_outputs[1][1:])

        # residual
        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]:
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else:
                    skip_states[i] = block(skip_states[i], time)

        # decoder
        input_dim = math.floor(skip_states[-1].shape[1] ** 0.5)
        decoder_output = self.decoder(
            skip_states[-1],
            (input_dim, input_dim),
            time=time,
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_output[0]

        if not return_dict:
            output = (sequence_output,) + decoder_output[1:] + encoder_outputs[1:]
            return output

        return ScOTOutput(
            loss=None,
            output=sequence_output,
            hidden_states=(
                decoder_output.hidden_states + encoder_outputs.hidden_states
                if output_hidden_states
                else None
            ),
            attentions=(
                decoder_output.attentions + encoder_outputs.attentions
                if output_attentions
                else None
            ),
            reshaped_hidden_states=(
                decoder_output.reshaped_hidden_states
                + encoder_outputs.reshaped_hidden_states
                if output_hidden_states
                else None
            ),
        )

class LSCOT(Physical2Regional2Physical):
    """
    LANO: Graph Neural Operator + SCOT(Pretrained) + Graph Neural Operator
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rigraph: RegionInteractionGraph,
                 variable_mesh: bool = False,
                 drop_edge: float = 0.0,
                 patch_size: int = 8,
                 gno_config: GNOConfig = GNOConfig(),
                 scot_config: SCOTConfig = SCOTConfig(),
                 regional_points: tuple = (64, 64)):
        nn.Module.__init__(self)
        self.drop_edge = drop_edge
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = gno_config.lifting_channels 
        self.patch_size = patch_size
        self.H = regional_points[0]
        self.W = regional_points[1]

        # Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, rigraph, gno_config)
        self.processor = self.init_processor(self.node_latent_size, scot_config)
        self.decoder = self.init_decoder(output_size, rigraph, variable_mesh, gno_config)
    
    def init_encoder(self, input_size, rigraph, gno_config):
        return GNOEncoder(
            in_channels = input_size,
            out_channels = self.node_latent_size,
            gno_config = gno_config
        )
    
    def init_processor(self, node_latent_size, config):
        # PATCH Embedding
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * self.node_latent_size,
                                      self.patch_size * self.patch_size * self.node_latent_size)
        
        self.positions = self.get_patch_positions()
        
        if config.pretrain_trained:
            processor = LSCOTProcessor.from_pretrained(config.model_type)
            print(f"Successfully load the pretrained model: {config.model_type}")
        else:
            processor = LSCOTProcessor(ScOTConfig(**shallow_asdict(config.args)))
            print(f"Initialize the model without parameters")

        
        processor_input_channles = processor.embeddings.patch_embeddings.projection.out_channels
        assert node_latent_size * (self.patch_size**2) == processor_input_channles, (
            f"node_latent_size ({node_latent_size}) doesn't align with the SCOTPocessor input channels {processor_input_channles}"
        )
        sequence_length = (self.H // self.patch_size) * (self.W // self.patch_size)
        assert sequence_length == 1024, (
            f"the expected sequent length for ScOT should be 1024, however, get the sequence length {sequence_length}."
        ) 
        
        # Drop the embeddings layer and recovery layer.
        processor.embeddings = None
        processor.patch_recovery = None

        return processor
    
    def init_decoder(self, output_size, rigraph, variable_mesh, gno_config):
        # Initialize the GNO decoder
        return GNODecoder(
            in_channels=self.node_latent_size,
            out_channels=output_size,
            gno_config=gno_config
        )

    def encode(self, graph: RegionInteractionGraph, pndata: torch.Tensor) -> torch.Tensor:
        # Apply GNO encoder
        encoded = self.encoder(graph, pndata)
        return encoded

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
        
        # Impose positional embedding
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)  # shape [num_patches, 2]
        pos_emb = self.compute_absolute_embeddings(pos, self.patch_size * self.patch_size * self.node_latent_size)
        rndata = rndata + pos_emb

        input_dimensions = (num_patches_H, num_patches_W)
        
        rndata = self.processor(
            embedding_output = rndata, # shape [batch_size, 1024, 48] for T, 
            input_dimensions = input_dimensions,
            time = condition,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = False
            )

        if isinstance(rndata, tuple):
            rndata = rndata[0]
        else:
            rndata = rndata.output 

        rndata = rndata.view(batch_size, H * W, C)

        return rndata

    def decode(self, graph: RegionInteractionGraph, rndata: torch.Tensor) -> torch.Tensor:
        # Apply GNO decoder
        decoded = self.decoder(graph, rndata)
        return decoded

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

