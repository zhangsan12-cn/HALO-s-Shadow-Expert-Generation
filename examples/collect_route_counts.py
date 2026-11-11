"""Convert synthetic top-k routes to the planner's two micro-batch inputs.

Run: python -m examples.collect_route_counts
There is no training framework or distributed process group in this example.
"""
import json

from shadow_planner import default_owner_map_for_shadow, split_topk_counts_by_rank


def main():
    # Each rank has [tokens, top_k] ids; -1 marks an unused top-k slot.
    topk = [
        [[0, 2], [1, 4], [2, 6], [3, 7], [0, -1]],
        [[2, 3], [0, 4], [1, 6], [2, 7], [3, -1]],
        [[4, 5], [0, 2], [1, 6], [4, 7], [5, -1]],
        [[6, 7], [0, 4], [1, 2], [3, 6], [7, -1]],
    ]
    mb0, mb1 = split_topk_counts_by_rank(topk, num_experts=8)
    print(json.dumps({
        "counts_mb0_by_rank": mb0,
        "counts_mb1_by_rank": mb1,
        "owner_of_expert": default_owner_map_for_shadow(4, 8),
    }, indent=2))


if __name__ == "__main__":
    main()
