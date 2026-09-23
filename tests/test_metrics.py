import math
import unittest

from shoprank.metrics import dcg_at_k, mean_ndcg_at_k, ndcg_at_k


class MetricTests(unittest.TestCase):
    def test_dcg_hand_computed(self):
        expected = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(dcg_at_k([3, 2, 1], 3), expected)

    def test_ideal_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1, 0], 4), 1.0)

    def test_bad_order_is_below_one(self):
        self.assertLess(ndcg_at_k([0, 1, 2, 3], 4), 1.0)

    def test_empty_relevance_is_zero(self):
        self.assertEqual(ndcg_at_k([], 10), 0.0)

    def test_queries_are_averaged_equally(self):
        self.assertAlmostEqual(mean_ndcg_at_k({"a": [3, 0], "b": [0, 3]}, 2),
                               (1 + ndcg_at_k([0, 3], 2)) / 2)


if __name__ == "__main__":
    unittest.main()

