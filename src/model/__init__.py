from typing import Tuple, Optional, Union
from dataclasses import dataclass
from omegaconf import OmegaConf
import math

from .rano import RANO
from .gino import GINO, GINOConfig
from .rfno import RFNO
from .lano import LANO
from .lano_batch import LANOBATCH
from .lscot import LSCOT

from .p2r2p import Physical2Regional2Physical
from ..graph import RegionInteractionGraph
from ..utils.dataclass import shallow_asdict

def init_model_from_rigraph(rigraph:RegionInteractionGraph, 
                            input_size:int, 
                            output_size:int, 
                            drop_edge:float = 0.0,
                            variable_mesh:bool = False,
                            model:str="rigno",
                            config:dataclass = None
                            )->Union[Physical2Regional2Physical, RANO, GINO]:
    
    supported_models = [
        "rigno",
        "rano",
        "gino",
        "rfno",
        "lano",
        "lano_batch",
        "lscot",
    ]

    assert model.lower() in supported_models, (
        f"model {model} not supported, only support {supported_models} "
    )
    # determine the regional_points
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
            config       = config.deepgnn
        )
    
    elif model.lower() == "rano":
        return RANO(
            input_size = input_size,
            output_size = output_size,
            rigraph = rigraph,
            drop_edge = drop_edge,
            variable_mesh= variable_mesh,
            gnnconfig = config.deepgnn,
            attnconfig = config.transformer,
            regional_points= regional_points
        )    
    
    elif model.lower() == "gino": #TODO: need to fix the GINO part!
        if config is None:
            config = ginoconfig
        return GINO(
            in_channels = input_size,
            out_channels = output_size,
            **shallow_asdict(config)
        )
    
    elif model.lower() == "rfno":
        return RFNO(
                input_size = input_size,
                output_size = output_size,
                rigraph = rigraph,
                fno_config = config.fno,
                regional_points = regional_points 
            )
    
    elif model.lower() == 'lano':
        return LANO(
            input_size=input_size,
            output_size=output_size,
            rigraph=rigraph,
            variable_mesh=variable_mesh,
            drop_edge=drop_edge,
            gno_config=config.gno,
            attn_config=config.transformer,
            regional_points=regional_points
        )
    
    elif model.lower() == 'lano_batch':
        return LANOBATCH(
            input_size=input_size,
            output_size=output_size,
            rigraph=rigraph,
            variable_mesh=variable_mesh,
            drop_edge=drop_edge,
            gno_config=config.gno,
            attn_config=config.transformer,
            regional_points=regional_points
        )
    
    elif model.lower() == 'lscot':
        # TODO: beta version, revise it in the future
        if config is None:
            return LSCOT(
                input_size = input_size,
                output_size= output_size,
                rigraph = rigraph,
            )
        else:
            gno_config = OmegaConf.merge(gno_struct, config.gno)
            # scot_config = OmegaConf.merge(scot_struct, config.scot)
            gno_config = OmegaConf.to_object(gno_config)
            # scot_config = OmegaConf.to_object(scot_config)
            return LSCOT(
                input_size = input_size, 
                output_size = output_size,
                rigraph = rigraph,
                variable_mesh=variable_mesh,
                drop_edge = drop_edge,
                gno_config=config.gno,
                scot_config=config.scot,
                regional_points=regional_points
                )
    
    else:
        raise ValueError(f"model {model} not supported currently!")

