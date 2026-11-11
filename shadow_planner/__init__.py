"""Standalone shadow-expert selection with topology-independent and NUMA models."""
from .api import evaluate_shadow_plan, plan_shadow_experts
from .models import (
    ShadowCountCostParams,
    ShadowNumaModelParams,
    ShadowTimeModelParams,
    default_owner_map_for_shadow,
    shadow_additional_cost_ms,
    shadow_additional_cost_numa_ms,
)
from .planner import (
    plan_shadow_sets_for_layer_counts_numa,
    plan_shadow_sets_for_layer_counts_pure_compute,
)
from .routing import counts_from_topk, split_topk_counts_by_rank

__version__ = "0.1.0"
__all__ = [
    "ShadowTimeModelParams", "ShadowCountCostParams", "ShadowNumaModelParams",
    "plan_shadow_experts", "evaluate_shadow_plan",
    "plan_shadow_sets_for_layer_counts_pure_compute",
    "plan_shadow_sets_for_layer_counts_numa", "default_owner_map_for_shadow",
    "shadow_additional_cost_ms", "shadow_additional_cost_numa_ms",
    "counts_from_topk", "split_topk_counts_by_rank",
]
