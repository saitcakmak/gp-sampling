from test.utils import BotorchTestCase
import torch
from botorch.models import SingleTaskGP
from gp_sampling.decoupled_samplers import decoupled_sampler
from gp_sampling.thompson_samplers import decoupled_ts, exact_ts


class TestDecoupledTS(BotorchTestCase):
    def test_decoupled_ts(self):
        model = SingleTaskGP(torch.rand(50, 35), torch.randn(50, 1))

        ps = decoupled_sampler(model=model, sample_shape=[5], num_basis=2**8)
        next_sample = decoupled_ts(ps, num_draws=49000, d=35)
        self.assertTrue(True)
