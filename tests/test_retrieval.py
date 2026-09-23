import unittest

from shoprank.pipeline import search_demo
from shoprank.retrieval import BM25, TfidfCosine


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = ["red running shoe", "usb phone charger", "blue office chair"]

    def test_bm25_exact_document_wins(self):
        scores = BM25(self.documents).scores("phone charger")
        self.assertEqual(max(range(len(scores)), key=scores.__getitem__), 1)

    def test_tfidf_exact_document_wins(self):
        scores = TfidfCosine(self.documents).scores("phone charger")
        self.assertEqual(max(range(len(scores)), key=scores.__getitem__), 1)

    def test_end_to_end_demo_search(self):
        results = search_demo("wireless gaming mouse", limit=3)
        self.assertEqual(results[0]["product_id"], "P10")
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
