import sys

sys.path.append("..")

import unittest
import torch
import math
from utils import CTCLoss
from torch.autograd import gradcheck


class TestCTCCriterion(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda")

    def test_fwd_trivial(self):
        T = 3
        N = 2
        labels = [[0, 0]]
        emissions = (
            torch.FloatTensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
            .view(1, T, N)
            .to(self.device)
        )
        log_probs = torch.log(emissions)
        fwd = CTCLoss(log_probs.to(self.device), labels, N - 1)
        self.assertAlmostEqual(fwd.item(), 0.0)

    def test_fwd(self):
        T = 3
        N = 4
        labels = [[1, 2]]
        emissions = torch.FloatTensor([1.0] * T * N).view(1, T, N).to(self.device)
        log_probs = torch.log(emissions)
        m = torch.nn.LogSoftmax(2)
        log_probs = m(log_probs)
        fwd = CTCLoss(log_probs.to(self.device), labels, N - 1)
        self.assertAlmostEqual(fwd.item(), -math.log(0.25 * 0.25 * 0.25 * 5))

    def test_fwd_bwd(self):
        T = 5
        N = 6
        labels = [[0, 1, 2, 1, 0]]
        # fmt: off
        emissions = torch.tensor((
            0.633766,  0.221185, 0.0917319, 0.0129757,  0.0142857,  0.0260553,
            0.111121,  0.588392, 0.278779,  0.0055756,  0.00569609, 0.010436,
            0.0357786, 0.633813, 0.321418,  0.00249248, 0.00272882, 0.0037688,
            0.0663296, 0.643849, 0.280111,  0.00283995, 0.0035545,  0.00331533,
            0.458235,  0.396634, 0.123377,  0.00648837, 0.00903441, 0.00623107,
            ),
            device=self.device,
            requires_grad=True,
        )
        # fmt: on
        log_emissions = torch.log(emissions.view(1, T, N))
        log_emissions.retain_grad()
        fwd = CTCLoss(torch.nn.functional.log_softmax(log_emissions, 2), labels, N - 1)
        self.assertAlmostEqual(fwd.item(), 3.34211, places=4)
        fwd.backward()
        # fmt: off
        expected_grad = torch.tensor((
            -0.366234, 0.221185,  0.0917319, 0.0129757,  0.0142857,  0.0260553,
            0.111121,  -0.411608, 0.278779,  0.0055756,  0.00569609, 0.010436,
            0.0357786, 0.633813,  -0.678582, 0.00249248, 0.00272882, 0.0037688,
            0.0663296, -0.356151, 0.280111,  0.00283995, 0.0035545,  0.00331533,
            -0.541765, 0.396634,  0.123377,  0.00648837, 0.00903441, 0.00623107,
            ),
            device=self.device,
        ).view(1, T, N)
        # fmt: on
        self.assertTrue(log_emissions.grad.allclose(expected_grad))

    @unittest.skip("Enable when gtn supports retain grad graph.")
    def test_jacobian(self):
        T = 20
        N = 15
        B = 5
        tgt = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1],
            [0, 2, 3],
            [0, 0, 0, 0, 0],
            [0, 4, 8, 12],
        ]

        def fn(input):
            return CTCLoss(input, tgt, N - 1)

        def fn_mean(input):
            return CTCLoss(input, tgt, N - 1, "mean")

        inputs = torch.randn(B, T, N, dtype=torch.float, device = self.device, requires_grad=True)
        self.assertTrue(gradcheck(fn, (inputs), eps=1e-2, rtol=1e-3,
                                  atol=1e-2))
        self.assertTrue(
            gradcheck(fn_mean, (inputs), eps=1e-2, rtol=1e-3, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
