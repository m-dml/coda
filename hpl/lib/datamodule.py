from dataclasses import dataclass
from typing import Any


@dataclass
class L96Dataset:
    _target_: str = "hpl.datamodule.DataLoader.L96Dataset"
    simulator: Any = None
    x_grid_size: int = 40
    y_grid_size: int = 10
    time_step: float = 0.01
    n_integration_steps: int = 500
    n_spin_up_steps: int = 300
    additional_noise_std: float = 0
    n_masked_per_step: int = 20
    mask_fill_value: int = 0
    save_dir: Any = None
    load_dir: Any = None
    rollout_length: int = 25
    window_length: int = 15


@dataclass
class L96InferenceDataset:
    _target_: str = "hpl.datamodule.DataLoader.L96InferenceDataset"
    window_length: int = 15


@dataclass
class L96DataModule:
    _target_: str = "hpl.datamodule.DataLoader.L96DataLoader"
    _recursive_: bool = False
    dataset: Any = None
    simulator: Any = None
    save_training_data_dir: Any = None
    load_training_data_dir: Any = None
    data_amount: int = 12000
    train_validation_split: float = 0.75
    shuffle_train: bool = True
    shuffle_valid: bool = False
    batch_size: int = 32
    drop_last_batch: bool = False
    num_workers: int = 10
    pin_memory: bool = False
