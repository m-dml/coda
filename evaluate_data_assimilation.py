import os.path
import pickle

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.random
from mdml_tools.utils import logging as mdml_logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, R2Score
from tqdm import tqdm

from hpl.lib.config import register_configs

register_configs()


@hydra.main(config_path="conf", config_name="config_evaluate.yaml", version_base="1.3")
def main(cfg: DictConfig):
    console_logger = mdml_logging.get_logger(__name__)
    console_logger.info(f"Hydra version: {hydra.__version__}")

    console_logger.info(f"Applying random seed: {cfg.random_seed}")
    torch.random.manual_seed(cfg.random_seed)
    pl.seed_everything(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console_logger.info(f"Using {device} device.")

        checkpoint_file = os.path.join(cfg.exp_base_dir, "logs/checkpoints/assimilation_network.ckpt")
        console_logger.info(f"Loading model from checkpoint file located in '{checkpoint_file}'.")
        model = torch.load(checkpoint_file, map_location=device)

        expetiment_config_file = os.path.join(cfg.exp_base_dir, ".hydra/config.yaml")
        console_logger.info(f"Loading experiment config from '{expetiment_config_file}'.")
        experiment_config = OmegaConf.load(expetiment_config_file)
        # parameters from experiment config file
        experiment_window_length = experiment_config.datamodule.dataset.window_length
        console_logger.info(f"Set window length to {experiment_window_length}.")

        simulator = hydra.utils.instantiate(cfg.simulator).to(device)

        rmse, r2score = [], []
        for i in tqdm(range(cfg.n_trials)):
            dataset = hydra.utils.instantiate(
                cfg.dataset,
                simulator=simulator,
                window_length=experiment_window_length,
            )
            dataloader = DataLoader(dataset, batch_size=len(dataset))

            with torch.no_grad():
                for sample in dataloader:
                    reconstruction = model.forward(sample).squeeze()
            ground_truth = dataset.ground_truth

            # estimate metrics
            rmse_func = MeanSquaredError(squared=False)
            r2score_func = R2Score(num_outputs=reconstruction.size(1))

            rmse.append(rmse_func(ground_truth, reconstruction))
            r2score.append(r2score_func(ground_truth, reconstruction))

        rmse = torch.tensor(rmse).to("cpu")
        r2score = torch.tensor(r2score).to("cpu")
        data = pd.DataFrame({"rmse": rmse, "r2score": r2score})
        with open("data_stats.pkl", "wb") as file:
            pickle.dump(data, file)

    except Exception as exception:
        console_logger.exception(f"Error occurred during main(): {exception}.")
        print("Error!", exception)
        raise exception


if __name__ == "__main__":
    main()
