import logging
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.models.L96Model import Lorenz96One, Lorenz96Two
from src.models.L96Model import RungeKutta4thOrder
from src.datamodule import Dataset4DVar


class L96DataLoader(pl.LightningDataModule):

    def __init__(self,
                 path: str = None,
                 chunk_size: int = 10,
                 window: Tuple[int] = None,
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
            logging.info(f"Loading observations from path: {path}")
            self.data, self.model_params = self.load_data()
        else:
            logging.info(f"Generationg default observations")
            self.data, self.model_params = self.generate_observations()

        self.chunk_size = chunk_size
        self.window = window
        self.train_split, self.val_split = train_split, val_split
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def load_data(self):
        data: dict = np.load(self.path, allow_pickle=True).item()
        x = data["x_obs"].unsqueeze(0)
        params = data["params"]
        return x, params

    def setup(self, stage=None):
        train_end = int(self.data.size(-1) * self.train_split)
        self.train = Dataset4DVar(self.data[..., :train_end], self.model_params, self.chunk_size, self.window)
        self.valid = Dataset4DVar(self.data[..., train_end:], self.model_params, self.chunk_size, self.window)

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
                              noise_sigma=1., part_missing=0.1, levels=1):
        params = {
            "f": f,
        }

        if levels == 1:
            x = self.simulate_lorenz96_one_level(dt, k, f, n_steps, spin_up_steps)
        elif levels == 2:
            x = self.simulate_lorenz96_two_levels(dt, k, j, f, c, h, b, n_steps, spin_up_steps)
            params["c"] = c
            params["h"] = h
            params["b"] = b
        else:
            raise NotImplementedError

        torch.save(x, "simulation_true.tensor")
        x = self.corrupt_simulation(x, noise_sigma, part_missing)
        torch.save(x, "observations.tensor")
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
        x[mask == 0] = np.nan
        return x
