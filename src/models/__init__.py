import omegaconf
import torch
import torch.nn as nn
import hydra
import logging


class ParametrizedModel(nn.Module):

    def __init__(self, model: omegaconf.DictConfig, parametrization: omegaconf.DictConfig):
        super().__init__()
        logging.info(f"Instantiating <{model._target_}>")
        self.model = hydra.utils.instantiate(model)
        logging.info(f"Instantiating <{parametrization._target_}>")
        self.parametrization = hydra.utils.instantiate(parametrization)

    def forward(self, *args, **kwargs):
        x = self.model.forward(*args, **kwargs)
        x += self.parametrization.forward(*args)
        return x


class SuperModel(nn.Module):

    def __init__(self, tendencies_model, solver, parametrization=None):
        super().__init__()
        if parametrization:
            model = ParametrizedModel(tendencies_model, parametrization)
        else:
            logging.info(f"Instantiating <{tendencies_model._target_}>")
            model = hydra.utils.instantiate(tendencies_model)
        logging.info(f"Instantiating <{solver._target_}>")
        self.model = hydra.utils.instantiate(solver, model=model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
