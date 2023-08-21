import os

import h5py
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf


def load_test_simulations(path: str, n_simulations: int, device: str = "cpu") -> torch.Tensor:
    """Load test simulations from a file.

    Args:
        path (str): path to the file containing the test simulations
        n_simulations (int): number of simulations to load from the file
        device (str): torch device to load the data to. default: "cpu"

    Returns:
        torch.Tensor: test simulations as torch tensor
    """
    with h5py.File(path, "r") as file:
        if n_simulations:
            if n_simulations > len(file["first_level"]):
                raise ValueError("Number of simulations is higher than are in provided file.")
            data = torch.from_numpy(file["first_level"][:n_simulations])
        else:
            data = torch.from_numpy(file["first_level"][:])
        data = data.to(device)
    return data


def load_data_assimilation_network(directory: str, device: str = "cpu") -> torch.nn.Module:
    """Load the data assimilation network from a checkpoint.

    Args:
        directory (str): directory where hydra saved the output of the experiment
        device (str): torch device to load the network to. default: "cpu"

    Returns:
        torch.nn.Module: data assimilation network
    """
    path_to_checkpoint = os.path.join(directory, "logs/checkpoints/assimilation_network.ckpt")
    model = torch.load(path_to_checkpoint, map_location=device)
    return model


def load_hydra_config(directory: str) -> DictConfig:
    """Load the hydra config from a directory.

    Args:
        directory (str): directory where hydra saved the output of the experiment

    Returns:
        DictConfig: config saved by hydra during the experiment
    """
    path_to_config = os.path.join(directory, ".hydra/config.yaml")
    config = OmegaConf.load(path_to_config)
    return config


def find_experiments_directories(directories: list[str]) -> list[str]:
    """Find all experiment directories in a list of directories.

    Args:
        directories (List[str]): list of directories to search for experiments

    Return:
        List[str]: list of experiment directories where output is stored
    """
    path_list = []
    for directory in directories:
        for root, dirs, _ in os.walk(directory):
            for subdir in dirs:
                if ".hydra" in subdir:
                    path = os.path.join(root, subdir)[:-7]
                    path_list.append(path)
    return path_list


def get_full_keys(config: DictConfig, separator: str = "/") -> list[str]:
    """Get list of full keys from nested DictConfig. Joining all keys from nested DictConfig using provided separator.

    Args:
        config (DictConfig): hydra experiment configuration dictionary
        separator (str): character used for joining keys of nested DictConfigs. Default is "/".

    Returns:
        List[str]: list containing all extracted keys from hydra config dictionary
    """
    long_keys_dictionary = []
    for key, value in config.items():
        if isinstance(value, DictConfig):
            new_keys = get_full_keys(value)
            for inner_key in new_keys:
                long_keys_dictionary.append(f"{key}{separator}{inner_key}")
        else:
            long_keys_dictionary.append(key)
    return long_keys_dictionary


def dictconfig_to_dataframe(config: DictConfig, full_keys: list[str] = None, separator: str = "/") -> pd.DataFrame:
    """Convert DictConfig to DataFrame.
    Args:
        config (DictConfig): hydra experiment configuration dictionary
        full_keys (List[str]): list of keys to be added to DataFrame. By default, it uses all keys from hydra config.
        separator (str): character used to parse keys full keys. Default is "/".

    Returns:
        pd.DataFrame: pandas DataFrame containing all requested items
    """
    if full_keys is None:
        full_keys = get_full_keys(config, separator)

    new_keys_dict = {}
    for key in full_keys:
        key_parts_list = key.split(separator)
        entry = config
        for the_key in key_parts_list:
            entry = entry[the_key]
        new_keys_dict[key] = [entry]
    return pd.DataFrame(new_keys_dict)
