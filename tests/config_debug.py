import sys
sys.path.append("../")

from src.trainer.utils.default_set import (
    SetUpConfig, GraphConfig , ModelConfig, DatasetConfig, OptimizerConfig, PathConfig)

from omegaconf import OmegaConf

json_config = OmegaConf.load("../config/tpl/config_lano_uvit.json")

setupconfig_struct = OmegaConf.structured(SetUpConfig)
graphconfig_struct = OmegaConf.structured(GraphConfig)
modelconfig_struct = OmegaConf.structured(ModelConfig)
datasetconfig_struct = OmegaConf.structured(DatasetConfig)
optimizerconfig_struct = OmegaConf.structured(OptimizerConfig)
pathconfig_struct = OmegaConf.structured(PathConfig)


setup_config = OmegaConf.merge(setupconfig_struct, json_config.setup)
graph_config = OmegaConf.merge(graphconfig_struct, json_config.graph)
model_config = OmegaConf.merge(modelconfig_struct, json_config.model)
dataset_config = OmegaConf.merge(datasetconfig_struct, json_config.dataset)
optimizer_config = OmegaConf.merge(optimizerconfig_struct, json_config.optimizer)
path_config = OmegaConf.merge(pathconfig_struct, json_config.path)
breakpoint()




