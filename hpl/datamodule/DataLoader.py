import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def preprocess_data(data: torch.Tensor):
    """ Preprocess observations
    Args:
        data (torch.Tensor): masked observations by nan values

    Returns (torch.Tensor):
        binary mask added as channel. nan values replaced by zeros
    """
    mask = torch.full(data.size(), 1)
    masked_indexes = np.isnan(data)
    mask[masked_indexes] = 0
    data[masked_indexes] = 0
    return torch.stack((data, mask), 0)


def slice_and_patch(data: torch.Tensor, start: int, end: int):
    """ Slice data and patch it with zeros if indexes are out of bounds
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
        parts.append(torch.zeros(size))
        start = 0
    parts.append(data[..., start:end])
    if end > data.size(0):
        size[0] = end - data.size(0)
        parts.append(torch.zeros(size))

    return torch.concat(parts, dim=0)


class L96Dataset(Dataset):
    """Basic Lorenz96 Dataset for Data Assimilation problem
    Args:
        data (data: torch.Tensor): observational data on L96 system
        chunk_size (int): number of time-steps between two initial conditions
        window_size (int, int): number of time-steps used for feed forward path
    """
    def __init__(self, data: torch.Tensor, chunk_size: int, window_size: (int, int)):
        self.data = data
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

    def __getitem__(self, item):
        """
        Returns (torch.Tensor, torch.Tensor):
            feed forward input [CH, T, X] and chunk of observations [T, X]
        """
        index = self.sampling_indexes[item]
        left, right = self.ics_chunks[index]
        rollout = slice_and_patch(self.data, left, right+1)
        feed_forward = torch.stack(
            (
                slice_and_patch(self.data, left-self.window[0], left+self.window[1]),
                slice_and_patch(self.data, right-self.window[0], right+self.window[1]),
            ),
            dim=0,
        )
        return feed_forward, rollout


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

    def _load_observations(self, path: str):
        data: dict = np.load(path, allow_pickle=True).item()
        x: torch.Tensor = data["x_obs"].unsqueeze(0)
        return x

    def setup(self, stage: str = "training"):
        x = self._load_observations(self.path)
        if self.training_split == 1:
            self.train = L96Dataset(x, self.chunk_size, self.window)
        else:
            train_split_end = self.training_split * len(x)
            self.train = L96Dataset(x[:train_split_end], self.chunk_size, self.window)
            self.valid = L96Dataset(x[train_split_end:], self.chunk_size, self.window)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader | None:
        if hasattr(self, "valid"):
            return DataLoader(
                self.valid,
                shuffle=self.shuffle_valid,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return None
