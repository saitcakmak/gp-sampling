from test.utils import BotorchTestCase
import torch
from botorch.models import SingleTaskGP
from gp_sampling.decoupled_samplers import decoupled_sampler
from gp_sampling.thompson_samplers import (
    decoupled_ts,
    exact_ts,
    continuous_decoupled_ts,
)


class TestDecoupledTS(BotorchTestCase):
    def setUp(self):
        num_train = 20
        self.dim = 3
        self.model = SingleTaskGP(
            torch.rand(num_train, self.dim), torch.randn(num_train, 1)
        )

    def test_decoupled_ts(self):
        num_samples = 5
        ps = decoupled_sampler(
            model=self.model,
            sample_shape=[num_samples],
            num_basis=64,
        )
        next_sample = decoupled_ts(ps, num_draws=1024, d=self.dim)
        self.assertEqual(next_sample.shape, torch.Size([num_samples, self.dim]))

    def test_continuous_ts(self):
        num_samples = 5
        ps = decoupled_sampler(
            model=self.model,
            sample_shape=[num_samples],
            num_basis=64,
        )
        next_sample = continuous_decoupled_ts(
            ps, num_restarts=4, raw_samples=32, d=self.dim
        )
        self.assertEqual(next_sample.shape, torch.Size([num_samples, self.dim]))

        # test that the samples actually maximize the sample paths
        test_X = torch.rand(256, self.dim)
        test_Y = ps(test_X)
        max_Y, _ = test_Y.max(dim=-2, keepdim=True)
        ts_Y = ps(next_sample.unsqueeze(-2))
        self.assertTrue(torch.all(ts_Y >= max_Y))
