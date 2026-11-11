import unittest

import torch

from shadow_planner import counts_from_topk, split_topk_counts_by_rank


class RouteCountTests(unittest.TestCase):
    def test_entries_and_empty_slots(self):
        self.assertEqual(counts_from_topk([[0, 1], [1, -1], [2, 2]], 3), [1, 2, 2])

    def test_two_micro_batches_split_on_token_dimension(self):
        first, second = split_topk_counts_by_rank([[[0, 1], [1, 2], [2, -1]]], 3)
        self.assertEqual(first, [[1, 1, 0]])
        self.assertEqual(second, [[0, 1, 2]])

    def test_empty_micro_batch(self):
        first, second = split_topk_counts_by_rank([torch.empty((0, 2), dtype=torch.long)], 3)
        self.assertEqual(first, [[0, 0, 0]])
        self.assertEqual(second, [[0, 0, 0]])

    def test_invalid_routes(self):
        for ids in ([[0, 3]], [[-2, 0]], [[0.5, 1.0]], [[True, False]], [0, 1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                counts_from_topk(ids, 3)
        with self.assertRaises(ValueError):
            split_topk_counts_by_rank([[[0, 1]]], 3, split_at=2)


if __name__ == "__main__":
    unittest.main()
