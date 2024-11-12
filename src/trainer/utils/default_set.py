from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

# model default config
from ...model.cmpt.attn import TransformerConfig
from ...model.cmpt.deepgnn import DeepGraphNNConfig
from ...model.cmpt.gno import GNOConfig
from ...model.cmpt.scot import SCOTConfig
from ...model.cmpt.fno import FNOConfig
from ..optimizers import OptimizerargsConfig

from omegaconf import OmegaConf

def merge_config(default_config_class, user_config):
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)

@dataclass
class SetUpConfig:
    seed: int = 42                                          
    device: str = "cuda:0"
    dtype: str = "torch.float32"
    trainer_name: str = "sequential"                                        # [static, static_unstruc, sequential]
    train: bool = True
    test: bool = False
    ckpt: bool = False
    use_variance_test: bool = False                                         # TODO needs to develop.
    measure_inf_time: bool = False                                          # TODO needs to be examined

# For the initialization of RegionInteractionGraph
@dataclass
class GraphConfig:
    periodic: bool = False                                                  
    sample_factor: float = 0.5
    overlap_factor_p2r: float = 1.0
    overlap_factor_r2p: float = 1.0
    regional_level: int = 1
    add_dummy_node: bool = False    
    with_additional_info: bool = True
    regional_points: tuple = (64, 64)
    fix_radius: Optional[float] = None                                      # Whether to use the fix radius for building graph

@dataclass
class ModelArgsConfig:
    gno: GNOConfig = field(default_factory=GNOConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    scot: SCOTConfig = field(default_factory=SCOTConfig)
    deepgnn: DeepGraphNNConfig = field(default_factory=DeepGraphNNConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)

@dataclass
class ModelConfig:
    name: str = "lano"
    drop_edge: float = 0.0
    use_conditional_norm: bool = False
    variable_mesh: bool = False
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig)

@dataclass
class DatasetConfig:
    name: str = "CE-Gauss"
    metaname: str = "rigno-unstructured/CE-Gauss"
    base_path: str = "/cluster/work/math/camlab-data/rigno-unstructured/"
    use_metadata_stats: bool = False
    sample_rate: float = 0.1
    use_sparse: bool = False                                                # Use full resolution for Poseidon Dataset
    train_size: int = 1024
    val_size: int = 128
    test_size: int = 256
    rand_dataset: bool = False                                              # Whether to randomize the sequence of loaded dataset
    max_time_diff: int = 14                                                 # Max time difference        
    use_time_norm: bool = True                                              # whether to use normalization for lead time and time_difference
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    metric: str = "final_step"
    predict_mode: str = "all"
    stepper_mode: str = "output"                                            # [output, residual, time_der]

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    args: OptimizerargsConfig = field(default_factory=OptimizerargsConfig)

@dataclass
class PathConfig:
    ckpt_path: str = ".ckpt/test/test.pt"
    loss_path: str = ".loss/test/test.png"
    result_path: str = ".result/test/test.png"
    database_path: str = ".database/test/test.csv"


