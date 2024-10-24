from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class SetUpConfig:
    seed: int = 42
    device: str = "cuda:0"
    dtype: str = "torch.float32"
    trainer_name: str = "sequential"
    train: bool = True
    test: bool = False
    ckpt: bool = False
    use_variance_test: bool = False
    measure_inf_time: bool = False


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

@dataclass
class DatasetConfig:
    name: str = "CE-Gauss"
    metaname: str = "rigno-unstructured/CE-Gauss"
    base_path: str = "/cluster/work/math/camlab-data/rigno-unstructured/"
    use_metadata_stats: bool = False
    train_size: int = 1024
    val_size: int = 128
    test_size: int = 256
    max_time_diff: int = 14
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    latent_queries: tuple = (64, 64)
    metric: str = "final_step"
    predict_mode: str = "all"
    stepper_mode: str = "output"
