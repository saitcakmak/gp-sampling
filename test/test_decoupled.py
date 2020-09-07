from test.utils import BotorchTestCase
from gp_sampling.decoupled_samplers import decoupled_sampler
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
import itertools


class TestDecoupledSampler(BotorchTestCase):
    def test_decoupled_sampler(self):
        # tests the sample shape to ensure everything is working
        for train_n, dim, test_n, sample_shape, num_basis, transform in [
            [5, 1, 3, [1], 1, None],
            [20, 5, 15, [3, 5, 6], 5, Standardize(m=1)],
        ]:
            model = SingleTaskGP(
                torch.rand(train_n, dim),
                torch.rand(train_n, 1),
                outcome_transform=transform,
            )
            sampler = decoupled_sampler(
                model=model, sample_shape=sample_shape, num_basis=num_basis
            )
            test_X = torch.rand(test_n, dim)
            out = sampler(test_X)
            self.assertTupleEqual((*sample_shape, test_n, 1), tuple(out.shape))
