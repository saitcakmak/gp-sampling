import torch
from torch import Tensor
from torch.nn import Module
from typing import List, Callable
from gp_sampling.basis_functions import Basis


class Sampler(Module):
    r"""Abstract base class for samplers."""
    pass


class CompositeSampler(Sampler):
    def __init__(
        self,
        join_rule: Callable,
        samplers: List[Sampler],
        mean_function: Callable = None,
    ) -> None:
        r"""
        The composite sampler facilitates sampling from multiple samplers via a join rule.

        Args:
            join_rule: A callable to join samplers from different samplers, e.g. sum
            samplers: A list of samplers to sample from
            mean_function: A mean function that is added to the samples
        """
        super().__init__()
        self.join_rule = join_rule
        self.samplers = samplers
        self.mean_function = mean_function

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        r"""
        Sample from samplers at `X` and return the joined output.

        Args:
            X: An `n x d` tensor of sampling locations
            **kwargs: kwargs to be passed on to the samplers

        Returns:
            An `n x d` tensor of samples
        """
        samples = [sampler(X, **kwargs) for sampler in self.samplers]
        outputs = self.join_rule(samples)
        if self.mean_function is not None:
            outputs = outputs + self.mean_function(X).unsqueeze(-1)
        return outputs

    def reset_random_variables(self, *args, **kwargs) -> None:
        r"""Resets the random variables used by the samplers to generate the samples"""
        for sampler in self.samplers:
            sampler.reset_random_variables(*args, **kwargs)


class BayesianLinearSampler(Sampler):
    def __init__(
        self,
        weights: Tensor,
        basis: Basis = None,
        mean_function: Callable = None,
        weight_initializer: Callable = None,
    ) -> None:
        """
        Base class for representing samples as a weighted sum of basis
        functions, i.e. $f(x) = basis(x) @ w$.

        Args:
            weights: A `sample_shape x model_batch_shape x 1 x num_basis`-dim tensor
                of weights for linear combination of the basis
            basis: The basis function, if None uses b(X) = X
            mean_function: The mean function
            weight_initializer: A callable for initializing the weights
        """
        super().__init__()
        self.weights = weights
        self.basis = basis
        self.mean_function = mean_function
        self.weight_initializer = weight_initializer

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        r"""

        Args:
            X: A `model_batch_shape x n x d` tensor of sampling locations
            **kwargs: kwargs to be passed on to the basis

        Returns:
            An `sample_shape x model_batch_shape x n x 1` tensor of samples
        """
        features = X if self.basis is None else self.basis(X, **kwargs)
        outputs = torch.matmul(
            features, self.weights.permute(*range(self.weights.dim() - 2), -1, -2)
        )
        if self.mean_function is not None:
            outputs = outputs + self.mean_function(X)
        return outputs

    def reset_random_variables(self, reset_basis: bool = True) -> None:
        r"""
        Resets the random variables used for sampling.

        Args:
            reset_basis: If true, the basis random variables are also reset.
        """
        if reset_basis:
            self.basis.reset_random_variables()
        new_weights = self.weight_initializer(
            shape=self.weights.shape, dtype=self.weights.dtype
        )
        self.weights = new_weights
