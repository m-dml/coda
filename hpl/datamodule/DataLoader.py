import os
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from mdml_tools.simulators.lorenz96 import L96SimulatorOneLevel, L96SimulatorTwoLevel
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def preprocess_data(data: torch.Tensor):
    """Preprocess observations
    Args:
        data (torch.Tensor): masked observations by nan values

    Returns (torch.Tensor):
        extract mask from observations
    """
    masked = torch.from_numpy(np.isnan(data.numpy()))
    data = torch.masked_fill(data, masked, 0)
    mask = torch.logical_not(masked)
    return data, mask


def slice_and_patch(data: torch.Tensor, start: int, end: int, use_bool: bool = False):
    """Slice data and patch it with zeros if indexes are out of bounds
    Args:
        data (torch.Tensor): data to slice with time dimension in the beginning
        start (int): start index of slice
        end (int): end index of slice (not inclusive)

    Returns (torch.Tensor):
        slice of date patched with zeros
    """
    size = list(data.size())
    parts = []
    if start < 0:
        size[0] = abs(start)
        patch = torch.full(size, False) if use_bool else torch.zeros(size)
        parts.append(patch)
        start = 0
    parts.append(data[start:end])
    if end > data.size(0):
        size[0] = end - data.size(0)
        patch = torch.full(size, False) if use_bool else torch.zeros(size)
        parts.append(patch)
    return torch.concat(parts, dim=0)


def pack_feed_forward_input(data: torch.Tensor, mask: torch.Tensor, index: int, window: (int, int)):
    _data = slice_and_patch(data, index - window[0], index + window[1])
    _mask = slice_and_patch(mask, index - window[0], index + window[1], use_bool=True)
    _mask = _mask.float()
    return torch.stack((_data, _mask), dim=0)


class L96Dataset(Dataset):
    """Basic Lorenz96 Dataset for training Data Assimilation parametrization or/and parametrization.

    Dataset samples two feed-forward passes and chunk of observations.

    Args:
        data (torch.Tensor): observational data on L96 system [Time, Space].
        mask (torch.Tensor): observational mask [Time, Space];
            tensor of boolean where False states for masked value.
        chunk_size (int): number of time-steps between two initial conditions.
        window_past_size (int): number of time-steps from past used to forward pass.
        window_future_size (int) number of time-steps from future used to forward pass.
    """

    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.tensor,
        chunk_size: int,
        window_past_size: int,
        window_future_size: int,
    ):
        if data.shape != mask.shape:
            raise ValueError("data and mask must have the same shape")
        self.data = data
        self.mask = mask
        self.chunk_size = chunk_size
        self.window: tuple[int, int] = (window_past_size, window_future_size)

        ics_indexes = torch.arange(0, len(data) - chunk_size + 1)
        self.ics_chunks = torch.stack(
            (
                ics_indexes,
                ics_indexes + chunk_size - 1,
            ),
            dim=-1,
        )
        self.sampling_indexes = torch.arange(len(self.ics_chunks))

    def __len__(self):
        return len(self.sampling_indexes)

    def __getitem__(self, index: int):
        """Returns (torch.Tensor, torch.Tensor):

        feed forward input [CH, T, X] and chunk of observations [T, X]
        """
        i = self.sampling_indexes[index]
        left, right = self.ics_chunks[i]

        sample = {
            "feedforward_left": pack_feed_forward_input(self.data, self.mask, left, self.window),
            "feedforward_right": pack_feed_forward_input(self.data, self.mask, right, self.window),
            "observations_data": slice_and_patch(self.data, left, right + 1),
            "observations_mask": slice_and_patch(self.mask, left, right + 1),
        }
        return sample


class L96InferenceDataset(Dataset):
    """Inference Lorenz96 Dataset for evaluation Data Assimilation parametrization.

    Dataset samples feed-forward pass.

     Args:
         data (torch.Tensor): observational data on L96 system [Time, Space].
         mask (torch.Tensor): observational mask [Time, Space];
             tensor of boolean where False states for masked value.
         window_past_size (int): number of time-steps from past used to forward pass.
         window_future_size (int) number of time-steps from future used to forward pass.
    """

    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.tensor,
        window_past_size: int,
        window_future_size: int,
    ):
        if data.shape != mask.shape:
            raise ValueError("data and mask must have the same shape")
        self.data = data
        self.mask = mask
        self.window: tuple[int, int] = (window_past_size, window_future_size)
        self.sampling_indexes = torch.arange(len(self.data))

    def __len__(self):
        return len(self.sampling_indexes)

    def __getitem__(self, item):
        index = self.sampling_indexes[item]
        return pack_feed_forward_input(self.data, self.mask, index, self.window)


class L96DataLoader(pl.LightningDataModule):
    def __int__(
        self,
        dataset: DictConfig,
        simulator: DictConfig,
        x_grid_size: int = 40,
        y_grid_size: int = 10,
        time_step: float = 0.01,
        n_integration_steps: int = 500,
        n_spin_up_steps: int = 300,
        additional_noise_mean: float = 0,
        additional_noise_std: float = 0,
        n_masked_per_time_step: int = 20,
        mask_fill_value: int = 0,
        save_training_data_dir: str = None,
        load_training_data_dir: str = None,
        train_validation_split: float = 0.75,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.simulator: Union[L96SimulatorOneLevel, L96SimulatorTwoLevel] = instantiate(simulator)
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.time_step = time_step
        self.n_integration_steps = n_integration_steps
        self.n_spin_up_steps = n_spin_up_steps
        self.additional_noise_mean = additional_noise_mean
        self.additional_noise_std = additional_noise_std
        self.n_masked_per_time_step = n_masked_per_time_step
        self.mask_fill_value = mask_fill_value
        self.save_training_data_dir = save_training_data_dir
        self.load_training_data_dir = load_training_data_dir
        self.train_validation_split = train_validation_split
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        # load/generate training data
        if self.load_training_data_dir:
            data, mask = self.load_data()
        else:
            simulation = self.generate_data()
            data, mask = self.corrupt_data(simulation)

        if self.train_validation_split == 1:
            self.train = instantiate(self.dataset, data=data, mask=mask)
        else:
            train_split_end = int(self.train_validation_split * len(data))
            self.train = instantiate(self.dataset, data=data[:train_split_end], mask=mask[:train_split_end])
            self.valid = instantiate(self.dataset, data=data[train_split_end:], mask=mask[train_split_end:])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            drop_last=self.drop_last_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.coallate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            shuffle=self.shuffle_valid,
            batch_size=self.batch_size,
            drop_last=self.drop_last_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.coallate,
        )

    def coallate(self, batch: list[dict]) -> dict[str : torch.Tensor]:
        dict_batch = {
            "feedforward_left": list(),
            "feedforward_right": list(),
            "observations_data": list(),
            "observations_mask": list(),
        }
        for batch_el in batch:
            for key, value in batch_el.items():
                if key in dict_batch.keys():
                    dict_batch[key].append(value)
                else:
                    raise KeyError(f"Key '{key}' is not supported")
        for key, value in dict_batch.items():
            if len(value) == 0:
                dict_batch[key] = None
            else:
                dict_batch[key] = torch.stack(value, dim=0)
        return dict_batch

    def load_data(self):
        observations = torch.load(os.path.join(self.load_training_data_dir, "x_observations.pt"))
        mask = torch.load(os.path.join(self.load_training_data_dir, "x_mask.pt"))

        if mask.size(0) != 1 and observations.size(0) != 1:
            raise ValueError("first dimension of observations and mask tensors must be equal to 1")
        if observations.size(1) != self.n_integration_steps and mask.size(1) != self.n_spin_up_steps:
            raise ValueError(f"time dimension doesn't match specified value: {self.n_integration_steps}")
        if observations.size(-1) != self.x_grid_size and mask.size(-1) != self.x_grid_size:
            raise ValueError(f"grid dimension doesn't match specified value: {self.x_grid_size}")
        return observations.squeeze(), mask.squeeze()

    def generate_data(self):
        time_array = torch.arange(
            0,
            self.time_step * (self.n_spin_up_steps + self.n_integration_steps),
            self.time_step,
        )
        x_shape = torch.Size((1, 1, self.x_grid_size))
        x_init = self.simulator.forcing * (0.5 + torch.randn(x_shape, device="cpu") * 1.0)
        if isinstance(self.simulator, L96SimulatorTwoLevel):
            x_init = x_init / torch.tensor([self.y_grid_size, 50]).max()
            y_shape = torch.Size((1, 1, self.x_grid_size, self.y_grid_size))
            y_init = self.simulator.forcing * (0.5 + torch.randn(y_shape, device="cpu") * 1.0)
            y_init = y_init / torch.tensor([self.y_grid_size, 50]).max()
            x_data, _ = self.simulator.integrate(time_array, (x_init, y_init))
        else:
            x_data = self.simulator.integrate(time_array, x_init)

        if self.save_training_data_dir:
            torch.save(x_data, os.path.join(self.save_training_data_dir, "x_true.pt"))
            torch.save(x_data, os.path.join(self.save_training_data_dir, "time_array.pt"))
        return x_data[:, self.n_spin_up_steps :, :]

    def corrupt_data(self, states: torch.Tensor):
        # additional noise
        noise = torch.normal(
            mean=self.additional_noise_mean, std=self.additional_noise_std, size=states.size(), device="cpu"
        )
        observations = states + noise
        # random mask
        sample = torch.rand(states.size(), device="cpu").topk(self.n_masked_per_time_step, dim=-1).indices
        mask = torch.zeros(states.size(), device="cpu", dtype=torch.bool)
        mask.scatter_(dim=-1, index=sample, value=True)

        observations = torch.masked_fill(observations, mask, value=self.mask_fill_value)
        mask_binary = torch.logical_not(mask).float()
        if self.save_training_data_dir:
            torch.save(noise, os.path.join(self.save_training_data_dir, "x_noise.pt"))
            torch.save(observations, os.path.join(self.save_training_data_dir, "x_observations.pt"))
            torch.save(mask_binary, os.path.join(self.save_training_data_dir, "x_mask.pt"))
        return observations, mask_binary
