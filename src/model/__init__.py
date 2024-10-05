from typing import Tuple, Optional, Union
from dataclasses import dataclass
from omegaconf import OmegaConf

from .rano import RANO
from .gino import GINO, GINOConfig
from .p2r2p import Physical2Regional2Physical
from .cmpt.deepgnn import DeepGraphNN,DeepGraphNNConfig
from .cmpt.attn import TransformerConfig
from .cmpt.message_passing import MessagePassingLayerConfig
from .cmpt.mlp import AugmentedMLPConfig
from ..graph import RegionInteractionGraph
from ..utils.dataclass import shallow_asdict
from omegaconf import OmegaConf

def init_model_from_rigraph(rigraph:RegionInteractionGraph, 
                            input_size:int, 
                            output_size:int, 
                            drop_edge:float = 0.0,
                            variable_mesh:bool = False,
                            model:str="rigno",
                            gnnconfig:DeepGraphNNConfig = DeepGraphNNConfig(),
                            attnconfig:TransformerConfig = TransformerConfig(),
                            ginoconfig:GINOConfig = GINOConfig(),
                            config:dataclass = None
                            )->Union[Physical2Regional2Physical, RANO, GINO]:
    
    assert model.lower() in ["rigno", "rano", "gino"], f"model {model} not supported, only support `rigno`, `gino`, `ragno`"
    
    deepgnn_struct = OmegaConf.structured(DeepGraphNNConfig)
    attn_struct = OmegaConf.structured(TransformerConfig)

    if model.lower() == "rigno":
        return Physical2Regional2Physical(
            input_size   = input_size, 
            output_size  = output_size,  
            rigraph      = rigraph,
            drop_edge    = drop_edge,
            variable_mesh= variable_mesh,
            config       = gnnconfig
        )
    elif model.lower() == "rano":
        if config is None:
            return RANO(
                input_size = input_size,
                output_size = output_size,
                rigraph = rigraph,
                drop_edge = drop_edge,
                variable_mesh= variable_mesh,
                gnnconfig = gnnconfig,
                attnconfig = attnconfig,
            )
        else:
            deepgnn_config = OmegaConf.merge(deepgnn_struct, config.deepgnn)
            attn_config = OmegaConf.merge(attn_struct, config.transformer)
            deepgnn_config = OmegaConf.to_object(deepgnn_config)
            attn_config = OmegaConf.to_object(attn_config)
            return RANO(
                input_size = input_size,
                output_size = output_size,
                rigraph = rigraph,
                drop_edge = drop_edge,
                variable_mesh= variable_mesh,
                gnnconfig = deepgnn_config,
                attnconfig = attn_config,
            )    
    elif model.lower() == "gino":
        if config is None:
            config = ginoconfig
        return GINO(
            in_channels = input_size,
            out_channels = output_size,
            **shallow_asdict(config)
        )
    else:
        raise ValueError(f"model {model} not supported, only support `rigno`, `gino`, `ragno`")

