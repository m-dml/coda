import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from src.models.L96Model import Lorenz96One, Lorenz96Two
from src.models.L96Model import RungeKutta4thOrder


class AssimilationDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 params: dict,
                 chunk_size: int,
                 ):
        super().__init__()
        self.data = data
        self.params = params
        self.mask = torch.full_like(data, fill_value=1)
        self.mask[self.data == 0] = 0
        self.data[self.mask == 0] = 0
        self.chunk_size = chunk_size
        index_tensor = (
            torch.arange(chunk_size, self.total_time_steps - 2*chunk_size, chunk_size),
            torch.arange(2*chunk_size, self.total_time_steps - chunk_size, chunk_size),
        )
        self.index_tensor = torch.stack(index_tensor, dim=-1)

    @property
    def half_chunk_size(self) -> int:
        return int(self.chunk_size / 2)

    @property
    def total_time_steps(self) -> int:
        return self.data.size(-1)

    def __len__(self):
        return len(self.index_tensor)

    def __getitem__(self, index):
        rollout_target, rollout_target_mask = self.get_rollout_targets(index)
        ff_inputs = self.get_feed_forward_inputs(index)
        return ff_inputs, rollout_target, rollout_target_mask, self.params

    def get_rollout_targets(self, index):
        ic, ic_next = self.index_tensor[index]
        chunk_size = self.chunk_size

        data = (self.data[..., ic : ic_next], self.data[..., ic_next : ic_next+chunk_size])
        data = torch.stack(data, dim=0)

        mask = (self.mask[..., ic: ic_next], self.mask[..., ic_next: ic_next + chunk_size])
        mask = torch.stack(mask, dim=0)
        return data, mask

    def get_feed_forward_inputs(self, index):
        ic, ic_next = self.index_tensor[index]
        ic_prev = ic - self.chunk_size
        half_size = self.half_chunk_size

        neighborhood_left = self.data[..., ic_prev: ic+half_size]
        neighborhood_right = self.data[..., ic: ic_next+half_size]
        mask_neighborhood_left = self.mask[..., ic_prev: ic+half_size]
        mask_neighborhood_right = self.mask[..., ic: ic_next+half_size]

        neighborhoods = (
            torch.cat((neighborhood_left, mask_neighborhood_left), dim=0),
            torch.cat((neighborhood_right, mask_neighborhood_right), dim=0),
        )
        return torch.stack(neighborhoods)


class L96DataLoader(pl.LightningDataModule):

    def __init__(self,
                 path: str = None,
                 chunk_size: int = 10,
                 train_split=0.7,
                 val_split=0.3,
                 batch_size=1,
                 shuffle_train=True,
                 shuffle_valid=False,
                 num_workers=0,
                 pin_memory=False,
                 ):
        super().__init__()
        self.path = path
        if path:
            self.data, self.model_params = self.load_data()
        else:
            self.data, self.model_params = self.generate_observations()
        self.chunk_size = chunk_size
        self.train_split, self.val_split = train_split, val_split
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def load_data(self):
        raise NotImplementedError

    def setup(self, stage=None):
        train_end = int(self.data.size(-1) * self.train_split)
        self.train = AssimilationDataset(self.data[..., :train_end], self.model_params, self.chunk_size)
        self.valid = AssimilationDataset(self.data[..., train_end:], self.model_params, self.chunk_size)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=self.shuffle_valid,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def generate_observations(self, dt=0.01, k=40, j=10, f=8, c=1, h=10, b=10, n_steps=500, spin_up_steps=500,
                              seed=1000, noise_sigma=1., part_missing=0.1, levels=1):
        torch.manual_seed(seed)
        params = {
            "f": f,
        }
        if levels == 1:
            x = self.simulate_lorenz96_one_level(dt, k, f, n_steps, spin_up_steps)
            x = x.unsqueeze(0)
        elif levels == 2:
            x = self.simulate_lorenz96_two_levels(dt, k, j, f, c, h, b, n_steps, spin_up_steps)
            x = x.unsqueeze(0)
            params["c"] = c
            params["h"] = h
            params["b"] = b
        else:
            raise NotImplementedError

        x = self.corrupt_simulation(x, noise_sigma, part_missing)
        return x, params

    @staticmethod
    def simulate_lorenz96_one_level(dt, k, f, n_steps, spin_up_steps):
        l96 = Lorenz96One()
        model = RungeKutta4thOrder(l96, dt=dt)

        # spin up
        x_init = f * (0.5 + torch.randn(torch.Size((1, 1, k,)), device='cpu') * 1.0)
        for i in range(spin_up_steps):
            x_init = model.forward(x_init, f=f)

        # # simulate
        x = torch.empty(torch.Size((*x_init.shape, n_steps + 1)))
        x[..., 0] = x_init
        for i in range(1, n_steps + 1):
            x[..., i] = model.forward(x[..., i - 1], f=f)
        x = x.squeeze()

        return x

    @staticmethod
    def simulate_lorenz96_two_levels(dt, k, j, f, c, h, b, n_steps, spin_up_steps):
        l96 = Lorenz96Two()
        model = RungeKutta4thOrder(l96, dt=dt)

        # spin up
        x_init = f * (0.5 + torch.randn(torch.Size((k,)), device='cpu') * 1.0) / torch.tensor([j, 50]).max()
        y_init = f * (0.5 + torch.randn(torch.Size((k, j)), device='cpu') * 1.0) / torch.tensor([j, 50]).max()
        for i in range(spin_up_steps):
            x_init, y_init = model.forward(x_init, y_init, f=f, h=h, c=c, b=b)

        # simulate
        x = torch.empty(torch.Size((k, n_steps + 1)))
        y = torch.empty(torch.Size((k, j, n_steps + 1)))
        x[:, 0], y[:, :, 0] = x_init, y_init
        for i in range(1, n_steps + 1):
            x[:, i], y[:, :, i] = model.forward(x[:, i - 1], y[:, :, i - 1], f=f, h=h, c=c, b=b)

        return x

    @staticmethod
    def corrupt_simulation(x, noise_sigma, part_missing):
        x += torch.normal(mean=0, std=noise_sigma, size=x.size())

        def elements_masked(n_elements):
            masked = int(n_elements * part_missing)
            if masked > n_elements:
                masked = n_elements
            return masked

        n_el = torch.prod(torch.tensor(x.size())).item()
        n_masked = elements_masked(n_el)
        mask = torch.full(size=(n_el,), fill_value=1.0, dtype=torch.float32)
        mask[:n_masked] = 0
        indexes = torch.randperm(n_el, device='cpu')
        mask = mask[indexes].reshape(x.size())

        return x * mask
