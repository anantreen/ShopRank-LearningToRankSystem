import unittest

from shoprank.statistics import holm_bonferroni, paired_bootstrap


class StatisticsTests(unittest.TestCase):
    def test_positive_paired_delta(self):
        result = paired_bootstrap({"a": .2, "b": .3}, {"a": .4, "b": .6}, iterations=500)
        self.assertAlmostEqual(result["delta"], .25)
        self.assertGreater(result["ci_low"], 0)

    def test_holm_preserves_original_order(self):
        result = holm_bonferroni([.04, .001, .02])
        self.assertAlmostEqual(result[1]["adjusted_p"], .003)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()

