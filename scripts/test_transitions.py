import gtn
import unittest
import copy
from build_transitions import count_ngrams, prune_ngrams, build_graph, add_blank_grams, add_self_loops


class TestTransitions(unittest.TestCase):

    def test_ngram_counts(self):

        lines = "abcdefg"
        tokens_to_idx = {l : e for e, l in enumerate(lines)}
        counts = count_ngrams([lines], 1, tokens_to_idx)
        unigrams = counts[0].most_common()
        expected_unigrams = [((i,), 1) for i in range(len(lines))]
        self.assertEqual(set(unigrams), set(expected_unigrams))

        lines = ["abab", "baba"]
        counts = count_ngrams(lines, 3, tokens_to_idx)
        unigrams = counts[0].most_common()
        bigrams = counts[1].most_common()
        trigrams = counts[2].most_common()
        expected_unigrams = [((0,), 4), ((1,), 4)]
        self.assertEqual(set(unigrams), set(expected_unigrams))
        expected_bigrams = [
            ((0, 1), 3), ((1, 0), 3), ((-1, 0), 1), ((-1, 1), 1)]
        self.assertEqual(set(bigrams), set(expected_bigrams))
        expected_trigrams = [
            ((0, 1, 0), 2), ((1, 0, 1), 2), ((-1, -1, 0), 1),
            ((-1, 0, 1), 1), ((-1, -1, 1), 1), ((-1, 1, 0), 1)]
        self.assertEqual(set(trigrams), set(expected_trigrams))

        pruned_ngrams = prune_ngrams(counts, [0, 1, 1])
        pruned_uni = [(0,), (1,)]
        self.assertEqual(set(pruned_uni), set(pruned_ngrams[0]))
        pruned_bi = [(0, 1), (1, 0)]
        self.assertEqual(set(pruned_bi), set(pruned_ngrams[1]))
        pruned_tri = [(0, 1, 0), (1, 0, 1)]
        self.assertEqual(set(pruned_tri), set(pruned_ngrams[2]))


    def test_graph_build(self):
        # unigram test case
        graph = build_graph([[(0,), (1,)]])
        expected = gtn.Graph()
        expected.add_node(True, True)
        expected.add_arc(0, 0, 0)
        expected.add_arc(0, 0, 1)
        self.assertTrue(gtn.isomorphic(graph, expected))

        # bigram test cases
        # fmt: off
        ngrams = [
            [(0,)],  # unigrams
            [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1)]  # bigrams
        ]
        # fmt: on
        with self.assertRaises(ValueError):
            build_graph(ngrams)

        # fmt: off
        ngrams = [
            [(0,), (1,)],  # unigrams
            [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1)]  # bigrams
        ]
        # fmt: on
        graph = build_graph(ngrams)
        expected = gtn.Graph()
        expected.add_node(True, True)  # <s>
        expected.add_node(False, True)  # 0
        expected.add_node(False, True)  # 1
        expected.add_node(False, True)  # back-off
        expected.add_arc(0, 1, 0)
        expected.add_arc(1, 2, 1)
        expected.add_arc(0, 2, 1)
        expected.add_arc(2, 1, 0)
        expected.add_arc(2, 2, 1)
        expected.add_arc(0, 3, gtn.epsilon)
        expected.add_arc(1, 3, gtn.epsilon)
        expected.add_arc(2, 3, gtn.epsilon)
        expected.add_arc(3, 1, 0)
        expected.add_arc(3, 2, 1)
        self.assertTrue(gtn.isomorphic(expected, graph))

        # trigram test case
        # fmt: off
        ngrams = [
            [(0,), (1,)],  # unigrams
            [(-1, 0), (0, 1), (1, 1)],  # bigrams
            [(-1, -1, 0), (-1, 0, 1), (0, 1, 1), (1, 1, 1)],  # trigrams
        ]
        # fmt: on
        graph = build_graph(ngrams)
        expected = gtn.Graph()
        expected.add_node(True, True)   # (<s>, <s>)
        expected.add_node(False, True)  # (<s>, 0)
        expected.add_node(False, True)  # (0, 1)
        expected.add_node(False, True)  # (1, 1)
        expected.add_node(False, True)  # <s>
        expected.add_node(False, True)  # 0
        expected.add_node(False, True)  # 1
        expected.add_node(False, True)  # unigram back-off
        # trigram arcs
        expected.add_arc(0, 1, 0)
        expected.add_arc(1, 2, 1)
        expected.add_arc(2, 3, 1)
        expected.add_arc(3, 3, 1)
        # bigram arcs
        expected.add_arc(4, 1, 0)
        expected.add_arc(5, 2, 1)
        expected.add_arc(6, 3, 1)
        # unigram arcs
        expected.add_arc(7, 5, 0)
        expected.add_arc(7, 6, 1)
        # back-off to bigram
        expected.add_arc(0, 4, gtn.epsilon)
        expected.add_arc(1, 5, gtn.epsilon)
        expected.add_arc(2, 6, gtn.epsilon)
        expected.add_arc(3, 6, gtn.epsilon)
         # back-off to unigram
        for i in range(7):
            expected.add_arc(i, 7, gtn.epsilon)
        self.assertTrue(gtn.isomorphic(expected, graph))

    def test_blank_build(self):
            # a b c c c b a
            grams = [
                [(0,), (1,), (2,)],
                [(-1, 0,), (0, 1), (1, 2,), (2, 1,), (1, 0,)],
                [(-1, 0, 1), (0, 1, 2), (1, 2, 2), (2, 2, 2), (2, 2, 1), (2, 1, 0)],
            ]
            optional_grams = add_blank_grams(copy.deepcopy(grams), 3, "optional")
            forced_grams = add_blank_grams(copy.deepcopy(grams), 3, "forced")
            # fmt: off
            expected_optional_grams = [
                [(0,), (1,), (2,), (3,)],
                [
                    (-1, 0,), (0, 1), (1, 2,), (2, 1,), (1, 0,), (-1, 3),
                    (0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)
                ],
                [
                    (-1, 0, 1), (0, 1, 2), (1, 2, 2), (2, 2, 2), (2, 2, 1), (2, 1, 0), 
                    (-1, 3, 0), (-1, 0, 3), (0, 1, 3), (0, 3, 1), (1, 3, 2), (2, 3, 2), 
                    (2, 2, 3), (2, 3, 1), (2, 1, 3), (1, 3, 0), (1, 0, 3), (1, 2, 3), 
                    (3, 0, 3), (3, 1, 3), (3, 2, 3), (3, 0, 1), (3, 1, 2), (3, 2, 2), 
                    (3, 2, 1), (3, 1, 0),
                ],
            ]
            expected_forced_grams = [
                [(0,), (1,), (2,), (3,)],
                [(-1, 3), (0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)],
                [
                    (-1, 3, 0), (3, 0, 3), (0, 3, 1), (3, 1, 3), (3, 2, 3),
                    (2, 3, 2), (1, 3, 2), (2, 3, 1), (1, 3, 0),
                ],
            ]
            # fmt: on
            for a, b in [
                (optional_grams, expected_optional_grams),
                (forced_grams, expected_forced_grams),
            ]:
                self.assertEqual(len(a), len(b))
                for i in range(len(a)):
                    self.assertEqual(len(a[i]), len(b[i]))
                    self.assertEqual(set(a[i]), set(b[i]))


    def test_add_self_loops(self):
        # unigram test case
        ngrams = [[(0,), (1,), (2,)], [(0, 1,), (1, 2,)], [(0, 1, 2,)]]
        # fmt: off
        expected = [
            [(0,), (1,), (2,)], 
            [(0, 1), (1, 2), (0, 0), (1, 1), (2, 2)], 
            [(0, 1, 2), (0, 0, 1), (0, 1, 1), (1, 1, 2), (1, 2, 2), (0, 0, 0), (1, 1, 1), (2, 2, 2)]
        ]
        # fmt: on
        self.assertEqual(add_self_loops(ngrams) ,expected)
        
if __name__ == "__main__":
    unittest.main()
