import sys

sys.path.append("..")

import unittest
import utils


class PackReplabel(unittest.TestCase):
    def test_case(self):
        tokens = [0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5]
        rep0 = utils.pack_replabels(tokens, 0)
        self.assertEqual(rep0, [0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5])
        rep1 = utils.pack_replabels(tokens, 1)
        self.assertEqual(rep1, [1, 0, 2, 3, 0, 3, 4, 0, 4, 0, 5, 6, 0])
        rep2 = utils.pack_replabels(tokens, 2)
        self.assertEqual(rep2, [2, 0, 3, 4, 1, 5, 1, 5, 6, 7, 0])


class UnpackReplabel(unittest.TestCase):
    def test_case(self):
        tokens = [2, 0, 2, 3, 0, 3, 4, 0, 4, 0, 5, 6, 0]
        unrep0 = utils.unpack_replabels(tokens, 0)
        self.assertEqual(unrep0, [2, 0, 2, 3, 0, 3, 4, 0, 4, 0, 5, 6, 0])
        unrep1 = utils.unpack_replabels(tokens, 1)
        self.assertEqual(unrep1, [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5])
        unrep2 = utils.unpack_replabels(tokens, 2)
        self.assertEqual(unrep2, [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4])


if __name__ == "__main__":
    unittest.main()
