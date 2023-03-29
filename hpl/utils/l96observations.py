import numpy as np
import torch
from torchdiffeq import odeint

from hpl.model.lorenz96 import L96Simulator, Lorenz96Two


def simulate_l96_1level(
    f: float = 8,
    k: int = 40,
    dt: float = 0.01,
    n_steps: int = 500,
    spin_up_steps: int = 200,
    n_simulations: int = 1,
    method: str = "rk4",
    options: dict = None,
):
    """Generate 1 level Lorenz'96 simulation
    Args:
        f (float): Lorenz'96 model free parameter
        k (int): size of Lorenz'96 system state
        dt (float): time-step used to run simulation
        n_steps (int): number of simulation steps
        spin_up_steps (int): number of steps to spin up the model
        n_simulations (int): number of simulations in batch
        method (str): name of method from torchdiffeq
        options (dict): solver parameters

    Returns (torch.Tensor):
        simulation
    """
    model = L96Simulator(f=f, method=method, options=options)
    x_init = f * (0.5 + torch.randn(torch.Size((n_simulations, 1, k)), device="cpu") * 1.0)
    t = torch.arange(0, dt * (n_steps + spin_up_steps), dt)
    x_true = model.integrate(t, x_init)
    return x_true[:, spin_up_steps:, :]


def simulate_l96_2levels(
    f: float = 10,
    b: float = 10,
    c: float = 1,
    h: float = 10,
    k: int = 40,
    j: int = 10,
    dt: float = 0.01,
    n_steps: int = 500,
    spin_up_steps: int = 500,
    method: str = "rk4",
    options: dict = None,
):
    """Generate 2 level Lorenz'96 simulation
    Args:
        f (float): Lorenz'96 model free parameter
        b: (float | torch.Tensor | nn.Parameter): coupling parameter of Lorenz'96 model
        c: (float | torch.Tensor | nn.Parameter): coupling parameter of Lorenz'96 model
        h: (float | torch.Tensor | nn.Parameter): = coupling parameter of Lorenz'96 model
        k (int): size of Lorenz'96 the first level system state
        j (int): number of state point per one in the first level
        dt (float): time-step used to run simulation
        n_steps (int): number of simulation steps
        spin_up_steps (int): number of steps to spin up the model
        method (str): name of method from torchdiffeq
        options (dict): solver parameters

    Returns (torch.Tensor, torch.Tensor):
        tuple with two levels simulations
    """
    model = Lorenz96Two(f, b, c, h)
    x_init = f * (0.5 + torch.randn(torch.Size((k,)), device="cpu") * 1.0) / torch.tensor([j, 50]).max()
    y_init = f * (0.5 + torch.randn(torch.Size((k, j)), device="cpu") * 1.0) / torch.tensor([j, 50]).max()
    t = torch.arange(0, dt * (n_steps + spin_up_steps), dt)
    x, y = odeint(model, (x_init, y_init), t, method=method, options=options)
    return x[spin_up_steps:], y[spin_up_steps:]


def corrupt(data: torch.Tensor, sigma: float, missing: float):
    """Corrupt data by applying Gaussian noise and mask
    Args:
        data (torch.Tensor): data to be corrupted
        sigma (float): standard deviation of Gaussian noise
        missing (float): which part of the data should be masked

    Returns (torch.Tensor):
        observations
    """
    data += torch.normal(mean=0, std=sigma, size=data.size())

    def elements_masked(n_elements):
        masked = int(n_elements * missing)
        if masked > n_elements:
            masked = n_elements
        return masked

    # generating and applying mask
    n_el = torch.prod(torch.tensor(data.size())).item()
    n_masked = elements_masked(n_el)
    mask = torch.full(size=(n_el,), fill_value=1.0, dtype=torch.float32)
    mask[:n_masked] = 0
    indexes = torch.randperm(n_el, device="cpu")
    mask = mask[indexes].reshape(data.size())
    data[mask == 0] = np.nan
    return data


def generate_l96_1level_observations(
    f: float = 8,
    k: int = 40,
    dt: float = 0.01,
    n_steps: int = 500,
    spin_up_steps: int = 200,
    sigma: float = 0,
    missing: float = 0,
    random_seed: int = 100,
):
    torch.manual_seed(random_seed)
    ground_truth = simulate_l96_1level(f, k, dt, n_steps, spin_up_steps).squeeze()
    observations = corrupt(ground_truth.clone(), sigma, missing)
    data = {"x_true": ground_truth, "x_obs": observations}
    return data


def generate_l96_2level_observations(
    f: float = 10,
    b: float = 10,
    c: float = 1,
    h: float = 10,
    k: int = 40,
    j: int = 10,
    dt: float = 0.01,
    n_steps: int = 500,
    spin_up_steps: int = 500,
    sigma: float = 0,
    missing: float = 0,
    random_seed: int = 100,
):
    torch.manual_seed(random_seed)
    ground_truth, _ = simulate_l96_2levels(f, b, c, h, k, j, dt, n_steps, spin_up_steps)
    ground_truth = ground_truth.squeeze()
    observations = corrupt(ground_truth.clone(), sigma, missing)
    data = {"x_true": ground_truth, "x_obs": observations}
    return data
