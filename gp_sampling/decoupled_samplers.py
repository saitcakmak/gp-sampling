from typing import Union, List

import torch
from botorch.models import SingleTaskGP
from torch import Tensor
from gpytorch.lazy import delazify

from gp_sampling.base import CompositeSampler, BayesianLinearSampler
from gp_sampling.basis_functions import RandomFourierBasis, KernelBasis, Basis


def decoupled_sampler(
    model: SingleTaskGP,
    sample_shape: Union[list, tuple, torch.Size],
    num_basis: int,
    prior_basis: Basis = None,
) -> CompositeSampler:
    r"""
    Constructs the decoupled sampler as an instance of CompositeSampler.
    Corresponds to decoupled_sampler_gpr from the tf implementation.

    Args:
        model: The GP model
        sample_shape: Output shape of the sampler
        num_basis: Number of basis functions to use
        prior_basis: The choice of the prior basis

    Returns:
        A CompositeSampler object
    """

    def _create_prior_fn() -> BayesianLinearSampler:
        r"""
        Creates the prior function of the decoupled sampler

        Returns:
            The corresponding BayesianLinearSampler object
        """
        basis = prior_basis
        if basis is None:
            basis = RandomFourierBasis(kernel=model.covar_module, units=num_basis)

        def w_init(shape):
            return torch.randn(shape)

        weights = w_init(batch_shape + [num_basis])
        return BayesianLinearSampler(
            basis=basis, weights=weights, weight_initializer=w_init
        )

    def _create_update_fn() -> BayesianLinearSampler:
        r"""
        Creates the update part of the decoupled sampler

        Returns:
            The corresponding BayesianLinearSampler object
        """
        Z = model.train_inputs[0]
        u = model.train_targets
        # add jitter here if needed
        sigma2 = model.likelihood.noise
        if model.mean_module is not None:
            u = u - model.mean_module(Z)
        if hasattr(model, "outcome_transform") and model.outcome_transform is not None:
            u = model.outcome_transform.untransform(u)[0]
        u = u.unsqueeze(-1)
        m = Z.shape[-2]
        Kuu = delazify(model.covar_module(Z, Z))
        Suu = Kuu + torch.eye(m) * sigma2
        Luu = torch.cholesky(Suu)
        basis = KernelBasis(kernel=model.covar_module, centers=Z)

        def w_init(shape):
            r"""
            Initializes the weights via a Cholesky solve on the priors

            Args:
                shape: the shape of the weights

            Returns:
                weights of shape
            """
            prior_f = prior_fn(Z)
            prior_u = prior_f + (sigma2 ** 0.5) * torch.randn(
                prior_f.shape, dtype=prior_f.dtype
            )
            init = torch.cholesky_solve(u - prior_u, Luu)
            init = torch.conj(init).permute(*range(init.dim() - 2), -1, -2)
            assert tuple(init.shape) == tuple(shape)
            return init

        weights = w_init(shape=batch_shape + [m])
        return BayesianLinearSampler(
            basis=basis, weights=weights, weight_initializer=w_init
        )

    batch_shape = list(sample_shape) + [1]
    prior_fn = _create_prior_fn()
    update_fn = _create_update_fn()

    def list_add(tensors: List[Tensor]):
        return torch.add(*tensors)

    return CompositeSampler(
        join_rule=list_add,
        samplers=[prior_fn, update_fn],
        mean_function=model.mean_module,
    )
