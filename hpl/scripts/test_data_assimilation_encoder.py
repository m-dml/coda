import argparse
import os
import sys

# make hpl module visible
from pathlib import Path

import h5py
import numpy as np
import torch
from joblib import delayed, Parallel
from mdml_tools.utils import logging as mdml_logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from hpl.datamodule import L96InferenceDataset

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


def create_parser() -> argparse.ArgumentParser:
    """Creating the argument parser and adding the arguments."""
    parser = argparse.ArgumentParser(
        description="Make reconstructions from pseudo-observations generated from provided simulations using trained "
        "data assimilation encoder."
    )
    parser.add_argument("--output-dir", type=str, help="Path to the output directory.", required=True)
    parser.add_argument(
        "--experiment-dir", type=str, help="Path to the experiment directory where hydra saved results.", required=True
    )
    parser.add_argument("--data-path", type=str, help="Path to the h5 file containing simulations.", required=True)
    parser.add_argument("--seed", type=int, help="Seed to use for generating the mask and noise.", default=101)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    parser.add_argument("--n-simulations", type=int, help="Number of simulations to use from data.")
    parser.add_argument(
        "--noise-std", type=float, help="Standard deviation of Gaussian noise added to ground truth.", default=1.0
    )
    parser.add_argument(
        "--noise-std-min",
        type=float,
        help="Minimum standard deviation of Gaussian noise added to ground truth.",
        default=0.1,
    )
    parser.add_argument(
        "--noise-std-max",
        type=float,
        help="Maximum standard deviation of Gaussian noise added to ground truth.",
        default=3.0,
    )
    parser.add_argument(
        "--mask-fraction", type=float, help="Percentage of data points masked per time step.", default=0.75
    )
    parser.add_argument(
        "--mask-fraction-min", type=float, help="Minimum percentage of data points masked per time step.", default=0.1
    )
    parser.add_argument(
        "--mask-fraction-max", type=float, help="Maximum percentage of data points masked per time step.", default=0.9
    )
    parser.add_argument("--mesh-steps", type=int, help="Number of steps for mesh testing.", default=15)
    parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs.", default=1)
    parser.add_argument("--ignore-edges", action="store_true", help="Remove edge cases from dataset.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for generating the data. Can be any " "string that is accepted by torch.device().",
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Convert the arguments to a namespace and check if they are valid.

    Args:
        parser (argparse.ArgumentParser): The argument parser.

    Returns:
        (str): The parsed arguments.
    """
    logger = mdml_logging.get_logger()
    args = parser.parse_args()
    output_file = os.path.join(
        args.output_dir,
        f"lorenz96-sigma_{round(args.noise_std, 2)}-mask_{round(args.mask_fraction, 2)}-reconstruction.h5",
    )
    if os.path.exists(output_file):
        if args.overwrite:
            logger.warning(f"The output file {output_file} already exists and will be overwritten.")
        else:
            raise ValueError(
                f"The output directory {output_file} already exists. Please specify a different output directory "
                "or use the --overwrite flag."
            )
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    try:
        torch.device(args.device)
    except Exception as e:
        raise ValueError(f"Could not parse the device {args.device}.") from e

    return args


def load_simulations(args: argparse.Namespace) -> torch.Tensor:
    with h5py.File(args.data_path, "r") as file:
        if args.n_simulations:
            if args.n_simulations > len(file["first_level"]):
                raise ValueError("Number of simulations is higher than are in provided file.")
            data = torch.from_numpy(file["first_level"][: args.n_simulations])
        else:
            args.n_simulations = len(file["first_level"])
            data = torch.from_numpy(file["first_level"][:])
    return data


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    path_to_checkpoint = os.path.join(args.experiment_dir, "logs/checkpoints/assimilation_network.ckpt")
    model = torch.load(path_to_checkpoint, map_location=args.device)
    return model


def load_config(args: argparse.Namespace) -> DictConfig:
    path_to_config = os.path.join(args.experiment_dir, ".hydra/config.yaml")
    config = OmegaConf.load(path_to_config)
    return config


def test_single_case(args: argparse.Namespace, noise_std: float = None, mask_fraction: float = None, pbar: bool = True):
    logger = mdml_logging.get_logger()
    # Set the seed
    torch.manual_seed(args.seed)

    noise_std = noise_std if noise_std else args.noise_std
    mask_fraction = mask_fraction if mask_fraction else args.mask_fraction
    output_file = os.path.join(
        args.output_dir, f"lorenz96-sigma_{round(noise_std, 2)}-mask_{round(mask_fraction, 2)}-reconstruction.h5"
    )

    data = load_simulations(args)
    model = load_model(args)
    config = load_config(args)
    input_window_extend = config.datamodule.dataset.input_window_extend

    dataset = L96InferenceDataset(
        ground_truth_data=data,
        input_window_extend=input_window_extend,
        mask_fraction=mask_fraction,
        additional_noise_std=noise_std,
        drop_edge_samples=args.ignore_edges,
    )

    with h5py.File(output_file, "w") as f:
        logger.info("Saving Metadata to hdf5 file.")
        # save metadata:
        for k, v in vars(args).items():
            f.attrs[k] = v

        f.create_dataset(
            "reconstruction",
            shape=(args.n_simulations, len(dataset), data.size(-1)),
            dtype=np.float32,
        )

        # add dimensions to dataset:
        f["reconstruction"].dims[0].label = "simulation"
        f["reconstruction"].dims[1].label = "time"
        f["reconstruction"].dims[2].label = "grid"

        with torch.no_grad():
            iterator = enumerate(dataset)
            if pbar:
                iterator = tqdm(iterator, total=len(dataset))
            for i, sample in iterator:
                encoded_state = model.forward(sample)
                f["reconstruction"][:, i, :] = encoded_state.squeeze().cpu().numpy()


def test_mesh(args: argparse.Namespace):
    noise_std_values = torch.linspace(args.noise_std_min, args.noise_std_max, args.mesh_steps)
    mask_fraction_values = torch.linspace(args.mask_fraction_min, args.mask_fraction_max, args.mesh_steps)
    noise_std_mesh, mask_fraction_mesh = torch.meshgrid(noise_std_values, mask_fraction_values)
    settings = [(a.item(), b.item()) for a, b in zip(noise_std_mesh.flatten(), mask_fraction_mesh.flatten())]
    Parallel(n_jobs=args.n_jobs, verbose=True)(delayed(test_single_case)(args, a, b, False) for a, b in tqdm(settings))


if __name__ == "__main__":
    _parser = create_parser()
    parsed_args = parse_args(_parser)
    if parsed_args.n_jobs == 1:
        test_single_case(parsed_args)
    else:
        test_mesh(parsed_args)
