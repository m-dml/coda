from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset4DVar(Dataset):

    def __init__(
            self,
            data: torch.Tensor,
            params: dict,
            chunk_size: int = 3,
            window: Tuple[int, int] = (3, 3),
    ):
        mask = torch.full(data.size(), 1)
        mask[np.isnan(data)] = 0
        self.data = torch.stack((data, mask), 0)
        self.params = params
        self.chunk_size = chunk_size
        self.window = window

        self.first_chunk_index = torch.arange(0, self.data.size(-1)-chunk_size+1)

    def __len__(self):
        return len(self.first_chunk_index)

    def crop_and_patch(self, start: int, end: int):
        size = list(self.data.size())
        data = []

        if start < 0:
            size[-1] = abs(start)
            data.append(torch.zeros(size))
            start = 0

        data.append(self.data[..., start:end])

        if end > self.data.size(-1):
            size[-1] = end - self.data.size(-1)
            data.append(torch.zeros(size))

        return torch.concat(data, dim=-1)

    def get_feed_forward_input(self, iic: int):
        left_index = iic - self.window[0]
        right_index = iic + self.window[1]
        return self.crop_and_patch(left_index, right_index)

    def get_target(self, iic: int):
        left_index = iic
        right_index = iic + self.chunk_size
        return self.crop_and_patch(left_index, right_index)

    def __getitem__(self, index):
        iic = self.first_chunk_index[index]
        iic_next = iic + self.chunk_size - 1
        ff_input = [self.get_feed_forward_input(i) for i in [iic, iic_next]]
        ff_input = torch.stack(ff_input, 0)
        target = [self.get_target(i)  for i in [iic, iic_next]]
        target = torch.stack(target, 0)
        return ff_input, target
