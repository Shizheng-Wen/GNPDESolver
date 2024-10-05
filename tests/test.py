from dataclasses import dataclass, is_dataclass, field, fields
from typing import Any


def shallow_asdict(obj:Any)->dict:
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    return obj

@dataclass
class TestInner:
    a:int = 1
    b:int = 2

@dataclass
class Test:
    a:int = 1
    b:int = 2
    inner:TestInner = field(default_factory=TestInner)

t = Test()
print(shallow_asdict(t))
