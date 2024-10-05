from typing import Any
from dataclasses import is_dataclass, fields

def shallow_asdict(obj:Any)->dict:
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    return obj

def safe_replace(obj:Any, **kwargs)->Any:
    if is_dataclass(obj):
        for key, value in kwargs.items():
            if key in fields(obj):
                setattr(obj, key, value)
    return obj