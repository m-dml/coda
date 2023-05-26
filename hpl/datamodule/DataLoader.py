import os
from typing import Union

import hydra.utils
import pytorch_lightning as pl
import torch
import torch.nn.functional as tnf
from hydra.utils import instantiate
from mdml_tools.simulators.lorenz96 import L96SimulatorOneLevel, L96SimulatorTwoLevel
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class L96DatasetBase(Dataset):
    def __init__(
        self,
        simulator: Union[L96SimulatorOneLevel, L96SimulatorTwoLevel, None] = None,
        x_grid_size: int = 40,
        y_grid_size: int = 10,
        time_step: float = 0.01,
        n_integration_steps: int = 500,
        n_spin_up_steps: int = 300,
        additional_noise_std: float = 0,
        n_masked_per_step: int = 20,
        mask_fill_value: int = 0,
        save_dir: str = None,
        load_dir: str = None,
    ):
        self.simulator: Union[L96SimulatorOneLevel, L96SimulatorTwoLevel] = simulator
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.time_step = time_step
        self.n_integration_steps = n_integration_steps
        self.n_spin_up_steps = n_spin_up_steps
        self.additional_noise_std = additional_noise_std
        self.n_masked_per_step = n_masked_per_step
        self.mask_fill_value = mask_fill_value
        self.save_dir = save_dir
        self.load_dir = load_dir

        if load_dir:
            observations, mask = self.load_data()
        else:
            ground_truth = self.generate_simulation()
            observations, mask = self.corrupt_simulation(ground_truth)
        self.data = torch.stack((observations, mask), dim=1)
        self.n_time_steps = self.data.size(0)

    def load_data(self) -> (torch.Tensor, torch.Tensor):
        observations = torch.load(os.path.join(self.load_dir, "ground_truth.pt"))
        mask = torch.load(os.path.join(self.load_dir, "mask.pt"))
        return observations, mask

    def generate_simulation(self) -> torch.Tensor:
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
            ground_truth, _ = self.simulator.integrate(time_array, (x_init, y_init))
        else:
            ground_truth = self.simulator.integrate(time_array, x_init)
        ground_truth = ground_truth[:, self.n_spin_up_steps :, :].squeeze()

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            torch.save(ground_truth, os.path.join(self.save_dir, "ground_truth.pt"))
            torch.save(time_array, os.path.join(self.save_dir, "time.pt"))

        return ground_truth

    def corrupt_simulation(self, ground_truth: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        simulation_size = ground_truth.size()
        # additional noise
        noise = torch.normal(mean=0, std=self.additional_noise_std, size=simulation_size, device="cpu")
        observations = ground_truth + noise
        # random mask
        sample = torch.rand(simulation_size, device="cpu").topk(self.n_masked_per_step, dim=-1).indices
        mask = torch.zeros(simulation_size, device="cpu", dtype=torch.bool)
        mask.scatter_(dim=-1, index=sample, value=True)
        observations = torch.masked_fill(observations, mask, value=self.mask_fill_value)
        mask_inverse = torch.logical_not(mask).float()

        if self.save_dir:
            torch.save(noise, os.path.join(self.save_dir, "additional_noise.pt"))
            torch.save(observations, os.path.join(self.save_dir, "observations.pt"))
            torch.save(mask_inverse, os.path.join(self.save_dir, "mask.pt"))
        return observations, mask_inverse


class L96Dataset(L96DatasetBase):
    def __init__(self, rollout_length: int, window_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_length = rollout_length
        self.window_length = window_length

        self.data = self.data.movedim(0, -1)
        self.data = tnf.pad(self.data, (window_length, window_length + 1), mode="constant", value=0)
        self.data = self.data.movedim(-1, 1)

        self.rollout_starting_index = torch.arange(0, self.n_time_steps - rollout_length, step=1) + window_length

    def __len__(self):
        return len(self.rollout_starting_index)

    def __getitem__(self, index: int):
        start = self.rollout_starting_index[index]
        end = start + self.rollout_length + 1
        observations_data = self.data[0, start:end]
        observations_mask = self.data[1, start:end].bool()
        feed_forward_left = self.data[:, start - self.window_length : start + self.window_length + 1]
        feed_forward_right = self.data[:, end - self.window_length : end + self.window_length + 1]

        return observations_data, observations_mask, feed_forward_left, feed_forward_right


class L96InferenceDataset(L96DatasetBase):
    def __init__(self, window_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_length = window_length
        self.rollout_starting_index = torch.arange(0, self.n_time_steps, step=1) - self.window_length

    def __len__(self):
        return len(self.rollout_starting_index)

    def __getitem__(self, index: int):
        ic_index = self.rollout_starting_index[index]
        start = ic_index - self.window_length
        end = ic_index + self.window_length + 1
        return self.data[:, start:end]


class L96DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        simulator: DictConfig,
        save_training_data_dir: str = None,
        load_training_data_dir: str = None,
        data_amount: int = 12000,
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
        self.save_training_data_dir = save_training_data_dir
        self.load_training_data_dir = load_training_data_dir
        self.data_amount = data_amount
        self.train_validation_split = train_validation_split
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, **kwargs) -> None:
        self.train_data: Dataset = hydra.utils.instantiate(
            self.dataset,
            n_integration_steps=int(self.data_amount * self.train_validation_split),
            simulator=self.simulator,
            save_dir=os.path.join(self.save_training_data_dir, "train") if self.save_training_data_dir else None,
            load_dir=os.path.join(self.load_training_data_dir, "train") if self.load_training_data_dir else None,
        )
        self.valid_data: Dataset = hydra.utils.instantiate(
            self.dataset,
            n_integration_steps=int(self.data_amount * (1 - self.train_validation_split)),
            simulator=self.simulator,
            save_dir=os.path.join(self.save_training_data_dir, "valid") if self.save_training_data_dir else None,
            load_dir=os.path.join(self.load_training_data_dir, "valid") if self.load_training_data_dir else None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            drop_last=self.drop_last_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_data,
            shuffle=self.shuffle_valid,
            batch_size=self.batch_size,
            drop_last=self.drop_last_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
