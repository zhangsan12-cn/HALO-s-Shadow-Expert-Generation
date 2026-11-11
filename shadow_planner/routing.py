"""CPU input conversion and framework-independent top-k counting helpers."""
import operator
from typing import List, Optional, Sequence, Tuple

import torch


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or (
        isinstance(value, torch.Tensor) and value.dtype == torch.bool
    ):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return int(result)


def _counts_tensor_from_rows(
    counts_by_rank: Sequence[Optional[Sequence[int]]],
    num_ranks: int,
    num_experts: int,
) -> torch.Tensor:
    if len(counts_by_rank) != num_ranks:
        raise ValueError("Counts must contain one row per rank")
    out = torch.zeros((num_ranks, num_experts), dtype=torch.long, device="cpu")
    for rank, row in enumerate(counts_by_rank):
        if row is None:
            continue
        if len(row) != num_experts:
            raise ValueError(f"Rank {rank} must contain {num_experts} expert counts")
        values = [_integer(v, f"count[{rank}][{e}]") for e, v in enumerate(row)]
        # Sums are used by int64 PyTorch operations in the planner.
        if sum(values) >= 2**63:
            raise ValueError("Counts exceed the int64 range")
        out[rank] = torch.tensor(values, dtype=torch.long, device="cpu")
    if sum(int(row.sum().item()) for row in out) >= 2**63:
        raise ValueError("Total counts exceed the int64 range")
    return out


def _topk_tensor(expert_ids, num_experts: int) -> torch.Tensor:
    num_experts = _integer(num_experts, "num_experts", minimum=1)
    try:
        ids = torch.as_tensor(expert_ids).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("expert_ids must be a rectangular [tokens, top_k] integer array") from exc
    if ids.ndim != 2:
        raise ValueError("expert_ids must have shape [tokens, top_k]")
    if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
        raise ValueError("expert_ids must have an integer dtype")
    if ((ids < -1) | (ids >= num_experts)).any().item():
        raise ValueError("expert_ids must be -1 (empty) or a valid expert id")
    return ids.to(torch.long)


def counts_from_topk(expert_ids, num_experts: int) -> List[int]:
    """Count top-k route entries for one rank; -1 denotes an empty slot.

    Rows are tokens and columns are top-k slots. Every valid entry is counted;
    this helper does not simulate token deduplication or gate-weight filtering.
    """
    ids = _topk_tensor(expert_ids, num_experts).reshape(-1)
    return torch.bincount(ids[ids >= 0], minlength=num_experts).tolist()


def split_topk_counts_by_rank(
    topk_ids_by_rank: Sequence,
    num_experts: int,
    split_at: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Convert rank-indexed [tokens, top_k] arrays into two count matrices.

    By default each rank is split at floor(tokens / 2), matching the original
    two-micro-batch planner. An explicit split_at is applied to every rank.
    No process group or training framework is required.
    """
    if len(topk_ids_by_rank) == 0:
        raise ValueError("At least one rank is required")
    if split_at is not None:
        split_at = _integer(split_at, "split_at")
    first, second = [], []
    for expert_ids in topk_ids_by_rank:
        ids = _topk_tensor(expert_ids, num_experts)
        boundary = ids.shape[0] // 2 if split_at is None else split_at
        if boundary > ids.shape[0]:
            raise ValueError("split_at exceeds the number of tokens on a rank")
        first.append(counts_from_topk(ids[:boundary], num_experts))
        second.append(counts_from_topk(ids[boundary:], num_experts))
    return first, second
