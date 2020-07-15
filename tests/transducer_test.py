import sys

sys.path.append("..")

import gtn
import math
import torch
import unittest

from transducer import Transducer
from torch.autograd import gradcheck


class TestTransducer(unittest.TestCase):
    def test_fwd_trivial(self):
        T = 3
        N = 2
        emissions = torch.FloatTensor([1.0, 0.0, 0.0, 1.0, 1.0,
                                       0.0]).view(T, 1, N)
        log_probs = torch.log(emissions)

        # Check without blank:
        labels = [[0, 1, 0]]
        transducer = Transducer(
            tokens=["a", "b"], graphemes_to_idx={"a": 0, "b": 1})
        self.assertAlmostEqual(transducer(log_probs, labels).item(), 0.0)

        # Check with blank:
        labels = [[0, 0]]
        transducer = Transducer(
            tokens=["a"], graphemes_to_idx={"a": 0}, blank=True)
        self.assertAlmostEqual(transducer(log_probs, labels).item(), 0.0)

        # Check with repeats not allowed:
        labels = [[0, 0]]
        transducer = Transducer(
            tokens=["a"], graphemes_to_idx={"a": 0}, blank=True, allow_repeats=False)
        self.assertAlmostEqual(transducer(log_probs, labels).item(), 0.0)


    def test_fwd(self):
        T = 3
        N = 4
        labels = [[1, 2]]
        emissions = torch.FloatTensor([1.0] * T * N).view(T, 1, N)
        log_probs = torch.log(emissions)
        log_probs = torch.nn.functional.log_softmax(torch.log(emissions), 2)
        transducer = Transducer(
            tokens=["a", "b", "c"],
            graphemes_to_idx={"a" : 0, "b" : 1, "c": 2},
            blank=True)
        fwd = transducer(log_probs, labels)
        self.assertAlmostEqual(fwd.item(), -math.log(0.25 * 0.25 * 0.25 * 5))

    def test_fwd_bwd(self):
        T = 5
        N = 6
        labels = [[0, 1, 2, 1, 0]]
        emissions = torch.tensor(
            (
                0.633766,
                0.221185,
                0.0917319,
                0.0129757,
                0.0142857,
                0.0260553,
                0.111121,
                0.588392,
                0.278779,
                0.0055756,
                0.00569609,
                0.010436,
                0.0357786,
                0.633813,
                0.321418,
                0.00249248,
                0.00272882,
                0.0037688,
                0.0663296,
                0.643849,
                0.280111,
                0.00283995,
                0.0035545,
                0.00331533,
                0.458235,
                0.396634,
                0.123377,
                0.00648837,
                0.00903441,
                0.00623107,
            ),
            requires_grad=True,
        )
        log_emissions = torch.log(emissions.view(T, 1, N))
        log_probs = torch.nn.functional.log_softmax(log_emissions, 2)
        log_emissions.retain_grad()
        transducer = Transducer(
            tokens=["a", "b", "c", "d", "e"],
            graphemes_to_idx={"a" : 0, "b" : 1, "c": 2, "d":3, "e": 4},
            blank=True)

        loss = transducer(log_probs, labels)
        self.assertAlmostEqual(loss.item(), 3.34211, places=4)
        loss.backward()
        expected_grad = torch.tensor((
            -0.366234,
            0.221185,
            0.0917319,
            0.0129757,
            0.0142857,
            0.0260553,
            0.111121,
            -0.411608,
            0.278779,
            0.0055756,
            0.00569609,
            0.010436,
            0.0357786,
            0.633813,
            -0.678582,
            0.00249248,
            0.00272882,
            0.0037688,
            0.0663296,
            -0.356151,
            0.280111,
            0.00283995,
            0.0035545,
            0.00331533,
            -0.541765,
            0.396634,
            0.123377,
            0.00648837,
            0.00903441,
            0.00623107,
        )).view(T, 1, N)
        self.assertTrue(log_emissions.grad.allclose(expected_grad))


    def test_simple_decomposition(self):
        T = 5
        tokens = ["a", "b", "ab", "ba", "aba"]
        scores = torch.randn((T, 1, len(tokens)), requires_grad=True)
        labels = [[0, 1, 0]]
        transducer = Transducer(
            tokens=tokens, graphemes_to_idx={"a": 0, "b": 1})

        # Hand construct the alignment graph with all of the decompositions
        alignments = gtn.Graph(False)
        alignments.add_node(True)

        # Add the path ['a', 'b', 'a']
        alignments.add_node()
        alignments.add_arc(0, 1, 0)
        alignments.add_arc(1, 1, 0)
        alignments.add_node()
        alignments.add_arc(1, 2, 1)
        alignments.add_arc(2, 2, 1)
        alignments.add_node(False, True)
        alignments.add_arc(2, 3, 0)
        alignments.add_arc(3, 3, 0)

        # Add the path ['a', 'ba']
        alignments.add_node(False, True)
        alignments.add_arc(1, 4, 3)
        alignments.add_arc(4, 4, 3)

        # Add the path ['ab', 'a']
        alignments.add_node()
        alignments.add_arc(0, 5, 2)
        alignments.add_arc(5, 5, 2)
        alignments.add_arc(5, 3, 0)

        # Add the path ['aba']
        alignments.add_node(False, True)
        alignments.add_arc(0, 6, 4)
        alignments.add_arc(6, 6, 4)

        emissions = gtn.linear_graph(T, len(tokens), True)

        emissions.set_weights(scores.flatten().tolist())
        expected_loss = gtn.subtract(
            gtn.forward_score(emissions),
            gtn.forward_score(gtn.intersect(emissions, alignments)))

        loss = transducer(scores, labels)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), 5)
        loss.backward()
        gtn.backward(expected_loss)

        expected_grad = torch.tensor(emissions.grad().weights())
        expected_grad = expected_grad.view((T, 1, len(tokens)))
        self.assertTrue(torch.allclose(
            scores.grad, expected_grad, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
