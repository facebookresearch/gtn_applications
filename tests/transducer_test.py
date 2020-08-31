import sys

sys.path.append("..")

import gtn
import math
import torch
import unittest

import transducer
from transducer import Transducer, make_transitions_graph
from utils import CTCLoss
from torch.autograd import gradcheck


class TestConvTransducer(unittest.TestCase):
    def test_kernel_graph(self):
        def get_graph(l1, l2, add_skip=False):
            g = gtn.Graph()
            g.add_node(True)
            g.add_node(True)
            g.add_node()
            g.add_node(False, True)
            g.add_node(False, True)
            g.add_arc(0, 0, 2)
            g.add_arc(0, 1, l1)
            g.add_arc(1, 1, l1)
            g.add_arc(1, 2, 2)
            g.add_arc(2, 2, 2)
            g.add_arc(2, 3, l2)
            g.add_arc(3, 3, l2)
            g.add_arc(3, 4, 2)
            g.add_arc(4, 4, 2)
            if add_skip:
                g.add_arc(1, 3, l2)
            return g

        # Repeats with optional blank
        graph = transducer.make_kernel_graph([0, 0], 2, True)
        gtn.equal(graph, get_graph(0, 0, False))

        # No repeats without optional blank
        graph = transducer.make_kernel_graph([0, 1], 2, False)
        gtn.equal(graph, get_graph(0, 1, False))

        # No repeats with optional blank
        graph = transducer.make_kernel_graph([0, 1], 2, True)
        gtn.equal(graph, get_graph(0, 1, True))

    def test_fwd(self):
        lexicon = [(0, 0), (0, 1), (1, 0), (1, 1)]
        blank_idx = 2
        kernel_size = 5
        stride = 3
        convTrans = transducer.ConvTransduce1D(
            lexicon, kernel_size, stride, blank_idx)

        B = 2
        C = 3
        # Zero length inputs not allowed
        inputs = torch.randn(B, 0, C)
        with self.assertRaises(ValueError):
            convTrans(inputs)
        # Other inputs should be padded to be larger than kernel_size
        for Tin in [1, 2, 3, 4]:
            inputs = torch.randn(B, Tin, C)
            convTrans(inputs)

        Tin = (1, 3, 4, 6, 7, 8)
        Tout = (1, 1, 2, 2, 3, 3)
        for Ti, To in zip(Tin, Tout):
            inputs = torch.randn(B, Ti, C)
            outputs = convTrans(inputs)
            self.assertEqual(outputs.shape, (B, To, len(lexicon)))

    def test_bwd(self):
        lexicon = [(0, 0), (0, 1), (1, 0), (1, 1)]
        blank_idx = 2
        kernel_size = 5
        stride = 3
        convTrans = transducer.ConvTransduce1D(
            lexicon, kernel_size, stride, blank_idx)

        B = 2
        C = 3
        Tin = (1, 3, 4, 6, 7, 8)
        Tout = (1, 1, 2, 2, 3, 3)
        for Ti, To in zip(Tin, Tout):
            inputs = torch.randn(B, Ti, C, requires_grad=True)
            outputs = convTrans(inputs)
            outputs.backward(torch.ones_like(outputs))


class TestTransducer(unittest.TestCase):

    def test_fwd_trivial(self):
        T = 3
        N = 2
        emissions = torch.FloatTensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).view(1, T, N)
        log_probs = torch.log(emissions)

        # Check without blank:
        labels = [[0, 1, 0]]
        transducer = Transducer(tokens=["a", "b"], graphemes_to_idx={"a": 0, "b": 1})
        self.assertAlmostEqual(transducer(log_probs, labels).item(), 0.0)

        # Check with blank:
        labels = [[0, 0]]
        transducer = Transducer(tokens=["a"], graphemes_to_idx={"a": 0}, blank="optional")
        self.assertAlmostEqual(transducer(log_probs, labels).item(), 0.0)

        # Check with repeats not allowed:
        labels = [[0, 0]]
        transducer = Transducer(
            tokens=["a"], graphemes_to_idx={"a": 0}, blank="optional", allow_repeats=False
        )
        self.assertAlmostEqual(transducer(log_probs, labels).item(), 0.0)

    def test_fwd(self):
        T = 3
        N = 4
        labels = [[1, 2]]
        emissions = torch.FloatTensor([1.0] * T * N).view(1, T, N)
        log_probs = torch.log(emissions)
        log_probs = torch.nn.functional.log_softmax(torch.log(emissions), 2)
        transducer = Transducer(
            tokens=["a", "b", "c"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2},
            blank="optional",
        )
        fwd = transducer(log_probs, labels)
        self.assertAlmostEqual(fwd.item(), -math.log(0.25 * 0.25 * 0.25 * 5))

    def test_ctc(self):
        T = 5
        N = 6

        # Test 1
        labels = [[0, 1, 2, 1, 0]]
        # fmt: off
        emissions = torch.tensor((
            0.633766,  0.221185, 0.0917319, 0.0129757,  0.0142857,  0.0260553,
            0.111121,  0.588392, 0.278779,  0.0055756,  0.00569609, 0.010436,
            0.0357786, 0.633813, 0.321418,  0.00249248, 0.00272882, 0.0037688,
            0.0663296, 0.643849, 0.280111,  0.00283995, 0.0035545,  0.00331533,
            0.458235,  0.396634, 0.123377,  0.00648837, 0.00903441, 0.00623107,
            ),
            requires_grad=True,
        )
        # fmt: on
        log_emissions = torch.log(emissions.view(1, T, N))
        log_emissions.retain_grad()
        transducer = Transducer(
            tokens=["a", "b", "c", "d", "e"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
            blank="optional",
        )

        loss = transducer(log_emissions, labels)
        self.assertAlmostEqual(loss.item(), 3.34211, places=4)
        loss.backward(retain_graph=True)
        # fmt: off
        expected_grad = torch.tensor((
            -0.366234, 0.221185,  0.0917319, 0.0129757,  0.0142857,  0.0260553,
            0.111121,  -0.411608, 0.278779,  0.0055756,  0.00569609, 0.010436,
            0.0357786, 0.633813,  -0.678582, 0.00249248, 0.00272882, 0.0037688,
            0.0663296, -0.356151, 0.280111,  0.00283995, 0.0035545,  0.00331533,
            -0.541765, 0.396634,  0.123377,  0.00648837, 0.00903441, 0.00623107,
        )).view(1, T, N)
        # fmt: on
        self.assertTrue(log_emissions.grad.allclose(expected_grad))

        # Test 2
        labels = [[0, 1, 1, 0]]
        # fmt: off
        emissions = torch.tensor((
            0.30176,  0.28562,  0.0831517, 0.0862751, 0.0816851, 0.161508,
            0.24082,  0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549,
            0.230246, 0.450868, 0.0389607, 0.038309,  0.0391602, 0.202456,
            0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345,
            0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046,
            ),
            requires_grad=True,
        )
        # fmt: on
        log_emissions = torch.log(emissions.view(1, T, N))
        log_emissions.retain_grad()
        transducer = Transducer(
            tokens=["a", "b", "c", "d", "e"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
            blank="optional",
            allow_repeats=False,
        )
        loss = transducer(log_emissions, labels)
        self.assertAlmostEqual(loss.item(), 5.42262, places=4)
        loss.backward()

        # fmt: off
        expected_grad = torch.tensor((
            -0.69824,  0.28562,   0.0831517, 0.0862751, 0.0816851, 0.161508,
            0.24082,   -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549,
            0.230246,  0.450868,  0.0389607, 0.038309,  0.0391602, -0.797544,
            0.280884,  -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345,
            -0.576714, 0.315517,  0.0338439, 0.0393744, 0.0339315, 0.154046,
        )).view(1, T, N)
        # fmt: on
        self.assertTrue(log_emissions.grad.allclose(expected_grad))

    def test_simple_decomposition(self):
        T = 5
        tokens = ["a", "b", "ab", "ba", "aba"]
        scores = torch.randn((1, T, len(tokens)), requires_grad=True)
        labels = [[0, 1, 0]]
        transducer = Transducer(tokens=tokens, graphemes_to_idx={"a": 0, "b": 1})

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

        emissions.set_weights(scores.data_ptr())
        expected_loss = gtn.subtract(
            gtn.forward_score(emissions),
            gtn.forward_score(gtn.intersect(emissions, alignments)),
        )

        loss = transducer(scores, labels)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
        loss.backward()
        gtn.backward(expected_loss)

        expected_grad = torch.tensor(emissions.grad().weights_to_numpy())
        expected_grad = expected_grad.view((1, T, len(tokens)))
        self.assertTrue(
            torch.allclose(scores.grad, expected_grad, rtol=1e-4, atol=1e-5)
        )

    def test_ctc_compare(self):
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

        tokens = list((t,) for t in range(N - 1))
        graphemes_to_idx = {t: t for t in range(N - 1)}
        inputs = torch.randn(B, T, N, dtype=torch.float, requires_grad=True)

        # With and without target length reduction:
        for reduction in ["none", "mean"]:
            transducer = Transducer(
                tokens=tokens,
                graphemes_to_idx=graphemes_to_idx,
                blank="optional",
                allow_repeats=False,
                reduction=reduction,
            )
            ctc_inputs = torch.nn.functional.log_softmax(inputs, 2)
            ctc_result = CTCLoss(ctc_inputs, tgt, N - 1, reduction)
            ctc_result.backward()
            ctc_grad = inputs.grad
            inputs.grad = None

            transducer_result = transducer(inputs, tgt)
            transducer_result.backward()
            transducer_grad = inputs.grad
            inputs.grad = None

            self.assertAlmostEqual(
                ctc_result.item(), transducer_result.item(), places=4
            )
            self.assertTrue(
                torch.allclose(ctc_grad, transducer_grad, rtol=1e-4, atol=1e-5)
            )

    def test_viterbi(self):
        T = 5
        N = 4
        B = 2

        # fmt: off
        emissions1 = torch.tensor((
            0, 4, 0, 1,
            0, 2, 1, 1,
            0, 0, 0, 2,
            0, 0, 0, 2,
            8, 0, 0, 2,
            ),
            dtype=torch.float,
        ).view(T, N)
        emissions2 = torch.tensor((
            0, 2, 1, 7,
            0, 2, 9, 1,
            0, 0, 0, 2,
            0, 0, 5, 2,
            1, 0, 0, 2,
            ),
            dtype=torch.float,
        ).view(T, N)
        # fmt: on

        # Test without blank:
        labels = [[1, 3, 0], [3, 2, 3, 2, 3]]
        transducer = Transducer(
            tokens=["a", "b", "c", "d"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
            blank="none",
        )
        emissions = torch.stack([emissions1, emissions2], dim=0)
        predictions = transducer.viterbi(emissions)
        self.assertEqual([p.tolist() for p in predictions], labels)

        # Test with blank without repeats:
        labels = [[1, 0], [2, 2]]
        transducer = Transducer(
            tokens=["a", "b", "c"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2},
            blank="optional",
            allow_repeats=False,
        )
        emissions = torch.stack([emissions1, emissions2], dim=0)
        predictions = transducer.viterbi(emissions)
        self.assertEqual([p.tolist() for p in predictions], labels)

    def test_transitions(self):
        num_tokens = 4

        # unigram
        transitions = make_transitions_graph(1, num_tokens)
        expected = gtn.Graph()
        expected.add_node(True, True)
        for i in range(num_tokens):
            expected.add_arc(0, 0, i)
        self.assertTrue(gtn.isomorphic(transitions, expected))

        # bigram
        transitions = make_transitions_graph(2, num_tokens)
        expected = gtn.Graph()
        expected.add_node(True, True)
        for i in range(num_tokens):
            expected.add_node(False, True)
            expected.add_arc(0, i+1, i)
        for i in range(num_tokens):
            for j in range(num_tokens):
                expected.add_arc(i+1, j+1, j)
        self.assertTrue(gtn.isomorphic(transitions, expected))

        # trigram
        transitions = make_transitions_graph(3, num_tokens)
        expected = gtn.Graph()
        expected.add_node(True, True)
        for i in range(num_tokens):
            expected.add_node(False, True)
            expected.add_arc(0, i + 1, i)
        for i in range(num_tokens):
            for j in range(num_tokens):
                expected.add_node(False, True)
                expected.add_arc(
                    i + 1,
                    num_tokens * i + j + num_tokens + 1,
                    j)
        for i in range(num_tokens):
            for j in range(num_tokens):
                for k in range(num_tokens):
                    expected.add_arc(
                        num_tokens * i + j + num_tokens + 1,
                        num_tokens * j + k + num_tokens + 1,
                        k)
        self.assertTrue(gtn.isomorphic(transitions, expected))

    def test_asg(self):
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
            requires_grad=True,
        )

        tokens = [(n,) for n in range(N)]
        graphemes_to_idx = {n: n for n in range(N)}
        transducer = Transducer(
            tokens=tokens,
            graphemes_to_idx=graphemes_to_idx,
            ngram=2)

        loss = transducer(emissions, labels)
        self.assertAlmostEqual(loss.item(), 7.47995, places=4)

        loss.backward()
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
            ).view(N, N)
            / B
        )
        trans_grad = transducer.transition_params.grad[N:].view(N, N).T
        self.assertTrue(trans_grad.allclose(expected_trans_grad, rtol=1e-03))

    def test_asg_viterbi(self):
        T = 4
        N = 4
        inputs = torch.tensor(
            [0, 0, 0, 7, 0, 5, 4, 3, 0, 5, 8, 5, 0, 5, 4, 3],
            dtype=torch.float32,
        ).view(1, T, N)
        transitions = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0],
            dtype=torch.float32,
        ).view(N + 1, N)
        expected_path = [3, 2, 1]
        tokens = [(n,) for n in range(N)]
        graphemes_to_idx = {n: n for n in range(N)}
        transducer = Transducer(
            tokens=tokens,
            graphemes_to_idx=graphemes_to_idx,
            ngram=2)
        transducer.transition_params.data = transitions
        path = transducer.viterbi(inputs)[0].tolist()
        self.assertTrue(path == expected_path)

    def test_backoff_transitions(self):
        transitions = gtn.load("trans_backoff_test.txt")
        T = 4
        N = 5
        inputs = torch.randn(1, T, N, dtype=torch.float, requires_grad=True)
        labels = [[0, 1, 0]]
        tokens = [(n,) for n in range(N)]
        graphemes_to_idx = {n: n for n in range(N)}
        transducer = Transducer(
            tokens=tokens,
            graphemes_to_idx=graphemes_to_idx,
            blank="optional",
            allow_repeats=False,
            transitions=transitions)
        loss = transducer(inputs, labels)
        loss.backward()
        trans_p = transducer.transition_params
        analytic_grad = trans_p.grad
        epsilon = 1e-3
        numerical_grad = []
        with torch.no_grad():
            for i in range(trans_p.numel()):
                transducer.transition_params.data[i] += epsilon
                loss_up = transducer(inputs, labels).item()
                transducer.transition_params.data[i] -= 2*epsilon
                loss_down = transducer(inputs, labels).item()
                numerical_grad.append((loss_up - loss_down) / (2 * epsilon))
                transducer.transition_params.data[i] += epsilon
        numerical_grad = torch.tensor(numerical_grad)
        self.assertTrue(torch.allclose(
            analytic_grad, numerical_grad, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
