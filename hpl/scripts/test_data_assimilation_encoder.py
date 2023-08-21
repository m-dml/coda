import argparse
import os

import h5py
import numpy as np
import submitit
import torch
from joblib import delayed, Parallel
from mdml_tools.utils import logging as mdml_logging
from omegaconf import DictConfig
from tqdm import tqdm

from hpl.datamodule import L96InferenceDataset
from hpl.utils.postprocessing import load_data_assimilation_network, load_hydra_config, load_test_simulations

# make hpl module visible


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
    parser.add_argument("--slurm-partition", type=str, help="Slurm partition to submit a job.", default="")
    parser.add_argument("--timeout-min", type=int, help="Slurm job parameter.", default=3600)

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


def load_test_data(args: argparse.Namespace) -> torch.Tensor:
    data = load_test_simulations(args.data_path, args.n_simulations)
    args.n_simulations = data.size(0)
    return data


def load_model_from_checkpoint(args: argparse.Namespace) -> torch.nn.Module:
    model = load_data_assimilation_network(args.experiment_dir, args.device)
    return model


def load_configuration_file(args: argparse.Namespace) -> DictConfig:
    config = load_hydra_config(args.experiment_dir)
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

    data = load_test_data(args)
    model = load_model_from_checkpoint(args)
    config = load_configuration_file(args)
    input_window_extend = config.datamodule.dataset.input_window_extend

    dataset = L96InferenceDataset(
        ground_truth_data=data,
        input_window_extend=input_window_extend,
        mask_fraction=mask_fraction,
        additional_noise_std=noise_std,
        drop_edge_samples=args.ignore_edges,
    )
    dataset.to(args.device)

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
    Parallel(n_jobs=args.n_jobs)(delayed(test_single_case)(args, a, b, False) for a, b in tqdm(settings))


if __name__ == "__main__":
    logger = mdml_logging.get_logger()

    logger.info("Instantiate parser")
    _parser = create_parser()
    logger.info("Parse arguments provided to the script")
    parsed_args = parse_args(_parser)

    if len(parsed_args.slurm_partition) > 0:
        executor = submitit.AutoExecutor(folder=parsed_args.output_dir)
        executor.update_parameters(timeout_min=parsed_args.timeout_min, slurm_partition=parsed_args.slurm_partition)
        if parsed_args.n_jobs == 1:
            job = executor.submit(test_single_case, parsed_args)
        else:
            job = executor.submit(test_mesh, parsed_args)
        logger.info("Job was submitted")
        output = job.result()
    else:
        if parsed_args.n_jobs == 1:
            test_single_case(parsed_args)
        else:
            test_mesh(parsed_args)
