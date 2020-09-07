import torch
from torch import Tensor
from botorch.utils import draw_sobol_samples
from gp_sampling.base import CompositeSampler
from botorch.models import SingleTaskGP
from typing import Optional, Union


def decoupled_ts(posterior_sampler: CompositeSampler, num_draws: int, d: int) -> Tensor:
    r"""
    Apply TS algorithm on a discretization of the space using num_draws samples.
    Uses samples from decoupled sampler.

    Args:
        posterior_sampler: A callable for sampling from the posterior GP - typically a
            decoupled sampler
        num_draws: Number of points used to discretize the space - a power of 2
            recommended
        d: dimension of the solution space

    Returns:
        `q x d` tensor of the maximizers, where `q` is the sample_shape of
            posterior_sampler
    """
    X = draw_sobol_samples(
        torch.tensor([[0.0], [1.0]]).repeat(1, d), n=num_draws, q=1
    ).squeeze(-2)
    with torch.no_grad():
        samples = posterior_sampler(X).squeeze(-1)
    arg_max = torch.argmax(samples, dim=-1)
    return X[arg_max]


def exact_ts(
    model: SingleTaskGP,
    num_draws: int,
    d: int,
    q: int = 1,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    r"""
    Apply TS algorithm on a discretization of the space using num_draws samples.
    Uses samples from the GP posterior.

    Args:
        model: A fitted GP model
        num_draws: Number of points used to discretize the space - a power of 2
            recommended
        d: dimension of the solution space
        q: number of parallel samples
        device: Option to specify cpu / gpu. If None, defaults to GPU if available.

    Returns:
        `q x d` tensor of the maximizers
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device(device) if isinstance(device, str) else device
    X = (
        draw_sobol_samples(torch.tensor([[0.0], [1.0]]).repeat(1, d), n=num_draws, q=1)
        .squeeze(-2)
        .to(device)
    )
    with torch.no_grad():
        samples = model.posterior(X).rsample(sample_shape=torch.Size([q])).squeeze(-1).t()
    arg_max = torch.argmax(samples, dim=-1)
    return X[arg_max]


def continuous_decoupled_ts(posterior_sampler: CompositeSampler, d=int) -> Tensor:
    r"""
    Apply TS algorithm on a sample function drawn using decoupled sampler. The sample
    function is continuously optimized to obtain the thompson sample

    Args:
        posterior_sampler: A callable for sampling from the posterior GP - typically a
            decoupled sampler
        d: dimension of the solution space

    Returns:
        `q x d` tensor of the maximizers, where `q` is the sample_shape of
            posterior_sampler
    """
    raise NotImplementedError
