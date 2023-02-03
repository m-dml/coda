import logging
from typing import Any

import git
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class LogGitHash(Callback):  # This is not a dataclass!!
    def __init__(self, git_hash: str = None):
        self.git_hash = git_hash
        self.console_logger = logging.getLogger(__name__)

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        repo_destination = HydraConfig.get().runtime.cwd
        repo = git.Repo(path=repo_destination, search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.git_hash = sha
        self.console_logger.info(f"Git hash: {sha}")
        with open("git_hash.txt", "w") as file:
            file.write(sha)
