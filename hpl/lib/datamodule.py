from dataclasses import dataclass
from typing import Any


@dataclass
class L96BaseDataset:
    _target_: str = "hpl.datamodule.DataLoader.L96BaseDataset"
    _recursive_: bool = False
    additional_noise_std: float = 1.0
    random_mask_fraction: float = 0.75
    mask_even_locations: bool = False
    mask_fill_value: float = 0.0
    path_to_save_data: Any = None


@dataclass
class L96TrainingDataset(L96BaseDataset):
    _target_: str = "hpl.datamodule.DataLoader.L96TrainingDataset"
    _recursive_: bool = False
    rollout_length: int = 1
    input_window_extend: Any = None
    extend_channels: bool = True


@dataclass
class L96InferenceDataset(L96BaseDataset):
    _target_: str = "hpl.datamodule.DataLoader.L96InferenceDataset"
    _recursive_: bool = False
    input_window_extend: int = 10
    extend_channels: bool = True
    drop_edge_samples: bool = True


@dataclass
class L96DataLoader:
    dataset: Any
    path_to_load_data: str
    path_to_save_data: Any = None
    train_validation_split: float = 0.75
    shuffle_train: bool = True
    shuffle_valid: bool = False
    batch_size: int = 1
    drop_last_batch: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    _target_: str = "hpl.datamodule.DataLoader.L96DataLoader"
    _recursive_: bool = False
