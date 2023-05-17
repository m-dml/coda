from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class L96Dataset:
    _target_: str = "hpl.datamodule.DataLoader.L96Dataset"
    data: Any = None
    mask: Any = None
    chunk_size: int = 25
    window_past_size: int = 15
    window_future_size: int = 15


@dataclass
class L96InferenceDataset:
    _target_: str = "hpl.datamodule.DataLoader.L96InferenceDataset"
    chunk_size: int = 25
    window_past_size: int = 15
    window_future_size: int = 15


@dataclass
class L96DataModule:
    _target_: str = "hpl.datamodule.DataLoader.L96DataLoader"
    _recursive_: bool = False
    x_grid_size: int = 40
    y_grid_size: int = 10
    time_step: float = 0.01
    n_integration_steps: int = 500
    n_spin_up_steps: int = 300
    additional_noise_mean: float = 0
    additional_noise_std: float = 0
    n_masked_per_time_step: int = 20
    mask_fill_value: Union[int, float] = 0
    save_training_data_dir: Optional[str] = None
    load_training_data_dir: Optional[str] = None
    train_validation_split: float = 0.75
    shuffle_train: bool = True
    shuffle_valid: bool = False
    batch_size: int = 1
    drop_last_batch: bool = False
    num_workers: int = 1
    pin_memory: bool = False
    dataset: Any = None
    simulator: Any = None
