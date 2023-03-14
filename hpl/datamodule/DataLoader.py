import pickle
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
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
    """Basic Lorenz96 Dataset for Data Assimilation problem
    Args:
        data (data: torch.Tensor): observational data on L96 system
        mask (data: torch.Tensor): observational mask
        chunk_size (int): number of time-steps between two initial conditions
        window_size (int, int): number of time-steps used for feed forward path
    """

    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.tensor,
        chunk_size: int,
        window_size: (int, int),
    ):
        if data.shape != mask.shape:
            raise ValueError("data and mask must have the same shape")
        self.data = data
        self.mask = mask
        self.chunk_size = chunk_size
        self.window = window_size

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
        observations_data = slice_and_patch(self.data, left, right + 1)
        observations_mask = slice_and_patch(self.mask, left, right + 1)

        item = {
            "feedforward_left": pack_feed_forward_input(self.data, self.mask, left, self.window),
            "feedforward_right": pack_feed_forward_input(self.data, self.mask, right, self.window),
            "observations_data": observations_data,
            "observations_mask": observations_mask,
        }
        return item


class L96InferenceDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.tensor,
        window_size: (int, int),
    ):
        if data.shape != mask.shape:
            raise ValueError("data and mask must have the same shape")
        self.data = data
        self.mask = mask
        self.window = window_size
        self.sampling_indexes = torch.arange(len(self.data))

    def __len__(self):
        return len(self.sampling_indexes)

    def __getitem__(self, item):
        index = self.sampling_indexes[item]
        return pack_feed_forward_input(self.data, self.mask, index, self.window)


class L96DataModule(pl.LightningModule):
    """DataModule handling L96 observations
    Args:
        path (str): path to file with L96 observations
        chunk_size (int): number of time-steps in chunk of observations
        window (int, int): number of time-steps used for feed forward path
        training_split (int): part of data used for training (rest is used for validation)
        batch_size (int): number of chunks of observations used for batch
        num_workers (int): number of DataLoader workers
        pin_memory (bool): weather to pin memory
    """

    def __init__(
        self,
        path: str,
        chunk_size: int,
        window: (int, int),
        training_split: float = 1.0,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.path = path
        self.chunk_size = chunk_size
        self.window = window
        if 0 > training_split > 1:
            raise ValueError("Training split should be in range from 0 to 1")
        self.training_split = training_split
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @staticmethod
    def _load_observations(path: str):
        with open(path, "rb") as file:
            data: dict = pickle.load(file)
        x: torch.Tensor = data["x_obs"]
        x, mask = preprocess_data(x)
        return x, mask

    def setup(self, **kwargs):
        x, mask = self._load_observations(self.path)
        if self.training_split == 1:
            self.train = L96Dataset(x, mask, self.chunk_size, self.window)
        else:
            train_split_end = int(self.training_split * len(x))
            self.train = L96Dataset(x[:train_split_end], mask[:train_split_end], self.chunk_size, self.window)
            self.valid = L96Dataset(x[train_split_end:], mask[train_split_end:], self.chunk_size, self.window)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, None]:
        if hasattr(self, "valid"):
            return DataLoader(
                self.valid,
                shuffle=self.shuffle_valid,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return None
