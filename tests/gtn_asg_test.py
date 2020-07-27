import sys

sys.path.append("..")

import unittest
import torch
import math
from utils import ASGLoss
from models import ASG
from torch.autograd import gradcheck


class TestASGCriterion(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda")

    def test_fwd_bwd(self):
        T = 5
        N = 6
        B = 3
        labels = [[2, 1, 5, 1, 3], [4, 3, 5], [3, 2, 2, 1]]
        emissions = torch.tensor(
            [
                [
                    [-0.4340, -0.0254, 0.3667, 0.4180, -0.3805, -0.1707],
                    [0.1060, 0.3631, -0.1122, -0.3825, -0.0031, -0.3801],
                    [0.0443, -0.3795, 0.3194, -0.3130, 0.0094, 0.1560],
                    [0.1252, 0.2877, 0.1997, -0.4554, 0.2774, -0.2526],
                    [-0.4001, -0.2402, 0.1295, 0.0172, 0.1805, -0.3299],
                ],
                [
                    [0.3298, -0.2259, -0.0959, 0.4909, 0.2996, -0.2543],
                    [-0.2863, 0.3239, -0.3988, 0.0732, -0.2107, -0.4739],
                    [-0.0906, 0.0480, -0.1301, 0.3975, -0.3317, -0.1967],
                    [0.4372, -0.2006, 0.0094, 0.3281, 0.1873, -0.2945],
                    [0.2399, 0.0320, -0.3768, -0.2849, -0.2248, 0.3186],
                ],
                [
                    [0.0225, -0.3867, -0.1929, -0.2904, -0.4958, -0.2533],
                    [0.4001, -0.1517, -0.2799, -0.2915, 0.4198, 0.4506],
                    [0.1446, -0.4753, -0.0711, 0.2876, -0.1851, -0.1066],
                    [0.2081, -0.1190, -0.3902, -0.1668, 0.1911, -0.2848],
                    [-0.3846, 0.1175, 0.1052, 0.2172, -0.0362, 0.3055],
                ],
            ],
            device=self.device,
            requires_grad=True,
        )
        emissions.retain_grad()
        transitions = torch.zeros((N + 1, N), device=self.device, requires_grad=True)
        fwd = ASGLoss(emissions, transitions, labels)
        self.assertAlmostEqual(fwd.item(), 7.47995, places=4)

        fwd.backward()
        expected_grad = torch.tensor(
            [
                [
                    [0.1060, 0.1595, -0.7639, 0.2485, 0.1118, 0.1380],
                    [0.1915, -0.7524, 0.1539, 0.1175, 0.1717, 0.1178],
                    [0.1738, 0.1137, 0.2288, 0.1216, 0.1678, -0.8057],
                    [0.1766, -0.7923, 0.1902, 0.0988, 0.2056, 0.1210],
                    [0.1212, 0.1422, 0.2059, -0.8160, 0.2166, 0.1300],
                ],
                [
                    [0.2029, 0.1164, 0.1325, 0.2383, -0.8032, 0.1131],
                    [0.1414, 0.2602, 0.1263, -0.3441, -0.3009, 0.1172],
                    [0.1557, 0.1788, 0.1496, -0.5498, 0.0140, 0.0516],
                    [0.2306, 0.1219, 0.1503, -0.4244, 0.1796, -0.2579],
                    [0.2149, 0.1745, 0.1160, 0.1271, 0.1350, -0.7675],
                ],
                [
                    [0.2195, 0.1458, 0.1770, -0.8395, 0.1307, 0.1666],
                    [0.2148, 0.1237, -0.6613, -0.1223, 0.2191, 0.2259],
                    [0.2002, 0.1077, -0.8386, 0.2310, 0.1440, 0.1557],
                    [0.2197, -0.1466, -0.5742, 0.1510, 0.2160, 0.1342],
                    [0.1050, -0.8265, 0.1714, 0.1917, 0.1488, 0.2094],
                ],
            ],
            device=self.device,
        )

        expected_grad = expected_grad / B
        self.assertTrue(emissions.grad.allclose(expected_grad, rtol=1e-03))
        expected_trans_grad = (
            torch.tensor(
                [
                    [0.3990, 0.3396, 0.3486, 0.3922, 0.3504, 0.3155],
                    [0.3666, 0.0116, -1.6678, 0.3737, 0.3361, -0.7152],
                    [0.3468, 0.3163, -1.1583, -0.6803, 0.3216, 0.2722],
                    [0.3694, -0.6688, 0.3047, -0.8531, -0.6571, 0.2870],
                    [0.3866, 0.3321, 0.3447, 0.3664, -0.2163, 0.3039],
                    [0.3640, -0.6943, 0.2988, -0.6722, 0.3215, -0.1860],
                ],
                device=self.device,
            ).view(N, N)
            / B
        )
        self.assertTrue(transitions.grad[1:].allclose(expected_trans_grad, rtol=1e-03))

    def test_viterbi(self):
        T = 4
        N = 3
        input_list = [0, 0, 0, 7, 0, 5, 4, 3, 0, 5, 8, 5, 0, 5, 4, 3]
        trans_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 0]
        expected_path = [2, 1, 0] # collapsed from [2, 1, 1, 0]
        rep = 1
        asg = ASG(N, rep)
        for param in asg.parameters():
            param.data = torch.tensor(
                trans_list, device=self.device, dtype=torch.float32
            ).view(N + rep + 1, N + rep)
        inputs = torch.tensor(input_list, device=self.device, dtype=torch.float32).view(
            1, T, N + rep
        )
        path = asg.viterbi(inputs)[0].tolist()
        self.assertEqual(len(expected_path), len(path))
        for i in range(0, len(path)):
            self.assertEqual(expected_path[i], path[i])
        self.assertTrue(path == expected_path)

    @unittest.skip("Enable when gtn supports retain grad graph.")
    def test_jacobian(self):
        T = 20
        N = 15
        B = 5
        tgt = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 3],
            [0, 2, 3],
            [0, 2, 4, 6, 8],
            [0, 1, 0, 1],
        ]

        def fn(inputs, transition):
            return ASGLoss(inputs, transition, tgt)

        def fn_mean(inputs, transition):
            return ASGLoss(inputs, transition, tgt, "mean")

        inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True)
        transitions = torch.randn(N + 1, N, dtype=torch.float, requires_grad=True)
        self.assertTrue(
            gradcheck(fn, (inputs, transitions), eps=1e-2, rtol=1e-3, atol=1e-2)
        )
        self.assertTrue(
            gradcheck(fn_mean, (inputs, transitions), eps=1e-2, rtol=1e-3, atol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
