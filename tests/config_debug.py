import sys
sys.path.append("../")

from src.trainer.utils.default_set import (
    SetUpConfig, GraphConfig , ModelArgsConfig, DatasetConfig, OptimizerConfig, PathConfig)

from omegaconf import OmegaConf

config = OmegaConf.load("config/tpl/config_lano_uvit.json")





