"""
Copyright (c) Meta Platforms, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys

sys.path.insert(1, "../gtn_applications")

import unittest
import torch
import math
from criterions import stc
from torch.autograd import gradcheck


class TestSTCCriterion(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda")

    def test_fwd_trivial(self):
        T = 3
        N = 2
        labels = [[1, 1]]
        emissions = (
            torch.FloatTensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            .view(T, 1, N)
            .to(self.device)
        )
        log_probs = torch.log(emissions)
        stc_crit = stc.STC(0, 1, 1, 1)
        fwd = stc_crit(log_probs.to(self.device), labels)
        self.assertAlmostEqual(fwd.item(), 0.0)

    def test_fwd(self):
        T = 3
        N = 4
        labels = [[1, 2]]
        emissions = torch.FloatTensor([1.0] * T * N).view(T, 1, N).to(self.device)
        log_probs = torch.log(emissions)
        m = torch.nn.LogSoftmax(2)
        log_probs = m(log_probs)
        stc_crit = stc.STC(0, 1, 1, 1, "none")
        fwd = stc_crit(log_probs.to(self.device), labels)
        # all possible ways of arranging "* 1 * 2 *" in 3 time steps
        # [0,2-3],1,2 ; 1,2,[0-1,3] ; 1,[0-3],2
        self.assertAlmostEqual(fwd.item(), -math.log(0.25 * 0.25 * (0.75 + 0.75 + 1)))


if __name__ == "__main__":
    unittest.main()
