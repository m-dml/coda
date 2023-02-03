from abc import abstractmethod

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchdiffeq import odeint


class Lorenz96One(nn.Module):
    """Lorenz' 96 one level tendencies
    Args:
        f (float | torch.Tensor | nn.Parameter): free parameter of Lorenz'96 model
    """

    def __init__(self, f: float | torch.Tensor | nn.Parameter = 8):
        super().__init__()
        self.f = f

    def __str__(self):
        return f"<Lorenz96One(F={self.f})>"

    def __repr__(self):
        return str(self)

    def forward(
        self,
        t: float | torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dx = torch.empty_like(x)
        # edge cases
        dx[..., 0] = x[..., -1] * (x[..., 1] - x[..., -2]) - x[..., 0] + self.f
        dx[..., 1] = x[..., 0] * (x[..., 2] - x[..., -1]) - x[..., 1] + self.f
        dx[..., -1] = x[..., -2] * (x[..., 0] - x[..., -3]) - x[..., -1] + self.f
        # the rest
        dx[..., 2:-1] = x[..., 1:-2] * (x[..., 3:] - x[..., :-3]) - x[..., 2:-1] + self.f
        return dx


class Lorenz96Two(torch.nn.Module):
    """Lorenz' 96 two levels tendencies
    Args:
        f (float | torch.Tensor | nn.Parameter): free parameter of Lorenz'96 model
        b: (float | torch.Tensor | nn.Parameter): coupling parameter of Lorenz'96 model
        c: (float | torch.Tensor | nn.Parameter): coupling parameter of Lorenz'96 model
        h: (float | torch.Tensor | nn.Parameter): = coupling parameter of Lorenz'96 model
    """

    def __init__(
        self,
        f: float | torch.Tensor | nn.Parameter = 10,
        b: float | torch.Tensor | nn.Parameter = 10,
        c: float | torch.Tensor | nn.Parameter = 1,
        h: float | torch.Tensor | nn.Parameter = 10,
    ):
        super().__init__()
        self.f = f
        self.b = b
        self.c = c
        self.h = h
        self.model_x = Lorenz96One(f)

    def __str__(self):
        return f"<Lorenz96Two(f={self.f}, b={self.b}, c={self.c}, h={self.h})>"

    def __repr__(self):
        return str(self)

    def forward(
        self,
        t: float | torch.Tensor | None,
        ics: (torch.Tensor, torch.Tensor),
    ) -> (torch.Tensor, torch.Tensor):
        x, y = ics
        j = y.size(-1)

        dy = torch.empty_like(y)
        # edge cases
        dy[..., 0] = self.c * (self.b * y[..., 1] * (y[..., -1] - y[..., 2]) - y[..., 0] + self.h * x[:] / j)
        dy[..., -1] = self.c * (self.b * y[..., 0] * (y[..., -2] - y[..., 1]) - y[..., -1] + self.h * x[:] / j)
        dy[..., -2] = self.c * (self.b * y[..., -1] * (y[..., -3] - y[..., 0]) - y[..., -2] + self.h * x[:] / j)
        dy[..., -3] = self.c * (self.b * y[..., -2] * (y[..., -4] - y[..., -1]) - y[..., -2] + self.h * x[:] / j)
        # the rest
        dy[..., 1:-2] = self.c * (
            self.b * y[..., 2:-1] * (y[..., 0:-3] - y[..., 3:]) - y[..., 1:-2] + self.h * x[..., None] / j
        )

        dx = self.model_x.forward(t, x) - self.h * self.c * y[..., :].mean()
        return dx, dy


class BaseSimulator(nn.Module):
    """Fully-Differentiable Abstract Simulator Use differentiable solvers from torchdiffeq. see
    https://github.com/rtqichen/torchdiffeq.

    Args:
       method (str): name of method from torchdiffeq
       options (Dict): solver parameters
    """

    def __init__(
        self,
        method: str,
        options: dict = None,
    ):
        super().__init__()
        self.method = method
        self.options = options

    def integrate(self, t: torch.Tensor, ics: tuple[torch.Tensor]):
        return odeint(self, ics, t, method=self.method, options=self.options)

    @abstractmethod
    def forward(self, t: torch.Tensor, ics: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        pass


class L96SimulatorNN(BaseSimulator):
    """1 level Lorenz' 96 simulator This simulator can be parametrized and trained.

    Args:
        f (float | torch.Tensor | nn.Parameter): free parameter of Lorenz'96 model
        network (DictConfig | nn.Module | None): network parametrization
        method (str): name of method from torchdiffeq
        options (dict): solver parameters
    """

    def __init__(
        self,
        f: float | torch.Tensor | nn.Parameter = 8,
        network: DictConfig | nn.Module | None = None,
        method: str = "rk4",
        options: dict = None,
    ):
        super().__init__(method, options)
        self.f = f
        self.model = Lorenz96One(f=f)
        if network is None or isinstance(network, nn.Module):
            self.network: nn.Module | None = network
        elif isinstance(network, DictConfig):
            self.network: nn.Module = hydra.utils.instantiate(network)

    def forward(self, t: torch.Tensor, ics: tuple[torch.Tensor]):
        x = ics[0]
        dx = torch.empty_like(x)
        dx = self.model.forward(t, x)
        if self.network:
            dx += self.network.forward(x)
        return dx


class L96Simulator(BaseSimulator):
    """Lorenz' 96 simulator This simulator can be parametrized and trained.

    Args:
        f (float | torch.Tensor | nn.Parameter): free parameter of Lorenz'96 model
        b: (float | torch.Tensor | nn.Parameter): coupling parameter of Lorenz'96 model
        c: (float | torch.Tensor | nn.Parameter): coupling parameter of Lorenz'96 model
        h: (float | torch.Tensor | nn.Parameter): = coupling parameter of Lorenz'96 model
        method (str): name of method from torchdiffeq
        options (dict): solver parameters
    """

    def __init__(
        self,
        f: float | torch.Tensor | nn.Parameter = 10,
        b: float | torch.Tensor | nn.Parameter = 10,
        c: float | torch.Tensor | nn.Parameter = 1,
        h: float | torch.Tensor | nn.Parameter = 10,
        method: str = "rk4",
        options: dict = None,
    ):
        super().__init__(method, options)
        self.f = f
        self.b = b
        self.c = c
        self.h = h
        self.model = Lorenz96Two(f, b, c, h)

    def forward(self, t: torch.Tensor, ics: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        diff = [torch.empty_like(el) for el in ics]
        diff = self.model.forward(t, ics)
        return tuple(diff)
