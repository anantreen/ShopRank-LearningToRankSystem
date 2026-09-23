import unittest

from shoprank.split import split_query_ids


class SplitTests(unittest.TestCase):
    def test_query_sets_are_disjoint_and_complete(self):
        ids = [f"q{i}" for i in range(100) for _ in range(3)]
        train, val, test = split_query_ids(ids)
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        self.assertEqual(train | val | test, set(ids))
        self.assertEqual((len(train), len(val), len(test)), (80, 10, 10))


if __name__ == "__main__":
    unittest.main()

