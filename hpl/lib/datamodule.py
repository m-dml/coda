from dataclasses import dataclass
from typing import Any


@dataclass
class L96BaseDataset:
    additional_noise_std: float
    mask_fraction: float
    mask_fill_value: float = 0.0
    path_to_save_data: str = None


@dataclass
class L96TrainingDataset(L96BaseDataset):
    rollout_length: int = 1
    input_window_extend: int = None
    extend_channels: bool = True


@dataclass
class L96InferenceDataset(L96BaseDataset):
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
