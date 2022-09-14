import logging

import torch
import torch.nn as nn


class Lorenz96One(nn.Module):
    device = None

    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                f: torch.Tensor,
                ) -> torch.Tensor:
        dx = torch.empty_like(x, device=self.device)
        # edge cases
        dx[:, :, 0] = x[:, :, -1] * (x[:, :, 1] - x[:, :, -2]) - x[:, :, 0] + f
        dx[:, :, 1] = x[:, :, 0] * (x[:, :, 2] - x[:, :, -1]) - x[:, :, 1] + f
        dx[:, :, -1] = x[:, :, -2] * (x[:, :, 0] - x[:, :, -3]) - x[:, :, -1] + f
        # the rest
        dx[:, :, 2:-1] = x[:, :, 1:-2] * (x[:, :, 3:] - x[:, :, :-3]) - x[:, :, 2:-1] + f
        return dx


class Lorenz96Two(nn.Module):
    device = None

    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                f: torch.Tensor,
                h: torch.Tensor,
                c: torch.Tensor,
                b: torch.Tensor,
                ) -> (torch.Tensor, torch.Tensor):
        # high frequency domain
        k = x.size(-1)
        j = y.size(-1)

        dy = torch.empty(torch.Size((k, j,)), device=self.device)
        # edge cases
        dy[:, 0] = c * (b * y[:, 1] * (y[:, -1] - y[:, 2]) - y[:, 0] + h * x[:] / j)
        dy[:, -1] = c * (b * y[:, 0] * (y[:, -2] - y[:, 1]) - y[:, -1] + h * x[:] / j)
        dy[:, -2] = c * (b * y[:, -1] * (y[:, -3] - y[:, 0]) - y[:, -2] + h * x[:] / j)
        dy[:, -3] = c * (b * y[:, -2] * (y[:, -4] - y[:, -1]) - y[:, -2] + h * x[:] / j)
        # the rest
        dy[:, 1:-2] = c * (b * y[:, 2:-1] * (y[:, 0:-3] - y[:, 3:]) - y[:, 1:-2] + h * x[:, None] / j)

        # low frequency domain
        dx = torch.empty(torch.Size((k,)), device=self.device)
        # edge cases
        dx[0] = x[-1] * (x[1] - x[-2]) - x[0] + f - h * c * y[0, :].mean()
        dx[1] = x[0] * (x[2] - x[-1]) - x[1] + f - h * c * y[1, :].mean()
        dx[-1] = x[-2] * (x[0] - x[-3]) - x[-1] + f - h * c * y[-1, :].mean()
        # the rest
        dx[2:-1] = x[1:-2] * (x[3:] - x[:-3]) - x[2:-1] + f - h * c * y[2:-1, :].mean(axis=1)
        return dx, dy


class RungeKutta4thOrder(nn.Module):

    def __init__(self, model: nn.Module, dt: float):
        super().__init__()
        self.add_module("model", model)
        self.dt = dt

    def forward(self, *args, **kwargs):
        """
        :param args: arguments those should be integrated
        :param kwargs: free parameters of model
        :return: updated arguments
        """
        f0 = self.model.forward(*args, **kwargs)
        f0_ = [args[i] + self.dt / 2 * x for i, x in enumerate(f0)]
        f1 = self.model.forward(*f0_, **kwargs)
        f1_ = [args[i] + self.dt / 2 * x for i, x in enumerate(f0)]
        f2 = self.model.forward(*f1_, **kwargs)
        f2_ = [args[i] + self.dt * x for i, x in enumerate(f0)]
        f3 = self.model.forward(*f2_, **kwargs)
        result = [x + self.dt / 6 * (f0[i] + 2 * (f1[i] + f2[i]) + f3[i]) for i, x in enumerate(args)]
        if len(result) == 1:
            return result.pop()
        return result


class L96AssimilationModel(nn.Module):

    def __init__(self, in_channels=2, input_size=(40, 15), activation=nn.ReLU()):
        super().__init__()

        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels=128, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding="same")
        in_features = torch.prod(torch.tensor(input_size)).item()
        self.linear = nn.Linear(in_features, input_size[0], bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = x.view((1, -1))
        x = self.linear(x)
        return x
