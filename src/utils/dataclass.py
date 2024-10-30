from typing import Any
from dataclasses import is_dataclass, fields
from omegaconf import DictConfig, OmegaConf

def shallow_asdict(obj:Any)->dict:
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    elif isinstance(obj, DictConfig):  
        return {k: v if not isinstance(v, DictConfig) else v for k, v in obj.items()}
    return obj

def safe_replace(obj:Any, **kwargs)->Any:
    if is_dataclass(obj):
        for key, value in kwargs.items():
            if key in fields(obj):
                setattr(obj, key, value)
    return obj