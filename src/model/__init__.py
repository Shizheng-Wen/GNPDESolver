from typing import Tuple, Optional, Union
from dataclasses import dataclass
from omegaconf import OmegaConf
import math

from .rano import RANO
from .gino import GINO, GINOConfig
from .rfno import RFNO
from .lano import LANO


from .p2r2p import Physical2Regional2Physical
from .cmpt.deepgnn import DeepGraphNN,DeepGraphNNConfig
from .cmpt.attn import TransformerConfig
from .cmpt.message_passing import MessagePassingLayerConfig
from .cmpt.mlp import AugmentedMLPConfig
from .cmpt.fno import FNOConfig
from .cmpt.gno import GNOConfig

from ..graph import RegionInteractionGraph
from ..utils.dataclass import shallow_asdict

def init_model_from_rigraph(rigraph:RegionInteractionGraph, 
                            input_size:int, 
                            output_size:int, 
                            drop_edge:float = 0.0,
                            variable_mesh:bool = False,
                            model:str="rigno",
                            gnnconfig:DeepGraphNNConfig = DeepGraphNNConfig(),
                            attnconfig:TransformerConfig = TransformerConfig(),
                            ginoconfig:GINOConfig = GINOConfig(),
                            gnoconfig: GNOConfig = GNOConfig(),
                            config:dataclass = None
                            )->Union[Physical2Regional2Physical, RANO, GINO]:
    
    assert model.lower() in ["rigno", "rano", "gino", "rfno", "lano"], f"model {model} not supported, only support `rigno`, `gino`, `rano`, 'rfno', 'lano'. "
    
    deepgnn_struct = OmegaConf.structured(DeepGraphNNConfig)
    attn_struct = OmegaConf.structured(TransformerConfig)
    fno_struct = OmegaConf.structured(FNOConfig)
    gno_struct = OmegaConf.structured(GNOConfig)

    num_nodes = rigraph.regional_to_regional.num_nodes
    sqrt_num_nodes = int(math.sqrt(num_nodes))
    regional_points = (sqrt_num_nodes, sqrt_num_nodes)
    
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
                regional_points= regional_points
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
                patch_size= config.patch_size,
                variable_mesh= variable_mesh,
                gnnconfig = deepgnn_config,
                attnconfig = attn_config,
                regional_points= regional_points
            )    
    elif model.lower() == "gino":
        if config is None:
            config = ginoconfig
        return GINO(
            in_channels = input_size,
            out_channels = output_size,
            **shallow_asdict(config)
        )
    elif model.lower() == "rfno":
        if config is None:
            return RFNO(
                input_size = input_size,
                output_size = output_size,
                rigraph = rigraph,
            )
        else:
            deepgnn_config = OmegaConf.merge(deepgnn_struct, config.deepgnn)
            fno_config = OmegaConf.merge(fno_struct, config.fno)
            deepgnn_config = OmegaConf.to_object(deepgnn_config)
            fno_config = OmegaConf.to_object(fno_config)
            return RFNO(
                    input_size = input_size,
                    output_size = output_size,
                    rigraph = rigraph,
                    fno_config = fno_config,
                    regional_points = regional_points 
                )
    
    elif model.lower() == 'lano':
        if config is None:
            return LANO(
                input_size = input_size,
                output_size = output_size,
                rigraph = rigraph,
            )
        else:
            gno_config = OmegaConf.merge(gno_struct, config.gno)
            attn_config = OmegaConf.merge(attn_struct, config.transformer)
            gno_config = OmegaConf.to_object(gno_config)
            attn_config = OmegaConf.to_object(attn_config)

            return LANO(
                input_size=input_size,
                output_size=output_size,
                rigraph=rigraph,
                gno_config=gno_config,
                attn_config=attn_config,
                regional_points=regional_points
            )

    else:
        raise ValueError(f"model {model} not supported, only support `rigno`, `gino`, `rano`, 'rfno', 'lano'")

