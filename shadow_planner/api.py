"""Public selection and independent plan-evaluation interfaces."""
from typing import Optional

from .models import (
    ShadowTimeModelParams, ShadowCountCostParams, ShadowNumaModelParams,
    _shadow_count_cost_params_from_env, _shadow_numa_params_from_env,
    _predict_shadow_T_and_gap_sum,
    shadow_additional_cost_ms, shadow_additional_cost_numa_ms,
)
from .planner import _plan_shadow_sets_for_layer_counts
from .routing import _counts_tensor_from_rows, _integer


def _topology(model: str, numa_params, num_ranks: int):
    if model == "cost":
        return None
    if model != "numa":
        raise ValueError("model must be 'cost' or 'numa'")
    topology = numa_params if numa_params is not None else _shadow_numa_params_from_env(num_ranks)
    if len(topology.rank_domains) != num_ranks:
        raise ValueError("rank_domains must have one entry per rank")
    return topology


def plan_shadow_experts(
    counts_mb0_by_rank,
    counts_mb1_by_rank,
    owner_of_expert,
    *,
    params: ShadowTimeModelParams,
    model: str = "cost",
    cost_params: Optional[ShadowCountCostParams] = None,
    numa_params: Optional[ShadowNumaModelParams] = None,
    M: int = 10,
    topL: int = 64,
    tau: float = 0.5,  # Starting threshold (ms); tune to measured time scales and noise.
    return_metrics: bool = False,
    exact_topk: int = 12,
    local_swap_rounds: int = 2,
    tau_rel: float = 0.0,
    fast_weights=(0.5, 0.3, 0.2),
):
    """Return CPU int64 shadow_ids [ranks, M] and optional diagnostic metrics.

    Each count matrix is [ranks, experts]. Entries count token/expert routes.
    -1 marks an unused output slot. Each (rank, expert) replica contributes once
    to the shadow communication cost, regardless of its token demand.

    model='cost' ignores topology; model='numa' weights inter-domain bandwidth.
    Explicit cost/topology parameters take precedence over environment defaults.
    The default tau is a starting point; tune it for the measured workload.
    """
    topology = _topology(model, numa_params, len(counts_mb0_by_rank))
    return _plan_shadow_sets_for_layer_counts(
        counts_mb0_by_rank, counts_mb1_by_rank, owner_of_expert,
        params=params, cost_params=cost_params, numa_params=topology,
        M=M, topL=topL, tau=tau, return_metrics=return_metrics,
        exact_topk=exact_topk, local_swap_rounds=local_swap_rounds,
        tau_rel=tau_rel, fast_weights=fast_weights,
    )


def evaluate_shadow_plan(
    counts_mb0_by_rank,
    counts_mb1_by_rank,
    owner_of_expert,
    shadow_ids,
    *,
    params: ShadowTimeModelParams,
    model: str = "cost",
    cost_params: Optional[ShadowCountCostParams] = None,
    numa_params: Optional[ShadowNumaModelParams] = None,
):
    """Rebuild traffic and evaluate an existing plan without running selection.

    This is useful for independently checking a returned plan and for evaluating
    both plans under one fixed model. Predictions are not measured training time.
    """
    ranks = len(counts_mb0_by_rank)
    if ranks == 0:
        raise ValueError("At least one rank is required")
    owner = [_integer(ow, "owner rank") for ow in owner_of_expert]
    if not owner or any(ow >= ranks for ow in owner):
        raise ValueError("owner_of_expert must contain valid owner ranks")
    if len(counts_mb1_by_rank) != ranks or len(shadow_ids) != ranks:
        raise ValueError("Counts and shadow_ids must contain one row per rank")
    topology = _topology(model, numa_params, ranks)
    costs = cost_params if cost_params is not None else _shadow_count_cost_params_from_env()
    selected = []
    width = len(shadow_ids[0])
    for rank, row in enumerate(shadow_ids):
        if len(row) != width:
            raise ValueError("shadow_ids must be rectangular [ranks, M]")
        values = [_integer(e, "shadow expert id", minimum=-1) for e in row]
        experts = [e for e in values if e >= 0]
        if len(experts) != len(set(experts)):
            raise ValueError("Duplicate shadow expert on a rank")
        if any(e >= len(owner) or owner[e] == rank for e in experts):
            raise ValueError("Shadow experts must exist and have a remote owner")
        selected.append(set(experts))

    intra = inter = 0
    for rank, experts in enumerate(selected):
        for expert in experts:
            if topology is not None and topology.rank_domains[rank] != topology.rank_domains[owner[expert]]:
                inter += 1
            else:
                intra += 1
    raw = []
    communication = []
    cross_rows = 0
    for counts in (counts_mb0_by_rank, counts_mb1_by_rank):
        matrix = _counts_tensor_from_rows(counts, ranks, len(owner)).tolist()
        total = [sum(row) for row in matrix]
        send, recv = [0] * ranks, [0] * ranks
        cross_send, cross_recv = [0] * ranks, [0] * ranks
        for rank, row in enumerate(matrix):
            for expert, count in enumerate(row):
                destination = owner[expert]
                if destination == rank or expert in selected[rank]:
                    continue
                send[rank] += count
                recv[destination] += count
                if topology is not None and topology.rank_domains[rank] != topology.rank_domains[destination]:
                    cross_send[rank] += count
                    cross_recv[destination] += count
                    cross_rows += count
        raw.extend((send, recv, total))
        extra = 0 if topology is None else 1 / topology.inter_bandwidth_ratio - 1
        communication.extend((
            [send[i] + cross_send[i] * extra for i in range(ranks)],
            [recv[i] + cross_recv[i] * extra for i in range(ranks)],
        ))
    halo, gap = _predict_shadow_T_and_gap_sum(
        *raw, params, communication_rows=tuple(communication) if topology is not None else None
    )
    additional = (
        shadow_additional_cost_ms(intra, costs) if topology is None
        else shadow_additional_cost_numa_ms(intra, inter, costs, topology)
    )
    result = {
        "planner_variant": model,
        "T_halo": halo,
        "gap": gap,
        "num_selected": intra + inter,
        "param_transfer_cost": additional / 2,
        "grad_sync_cost": additional / 2,
        "additional_cost": additional,
        "objective": halo + additional,
    }
    if topology is not None:
        result.update(
            num_selected_intra=intra, num_selected_inter=inter,
            cross_domain_token_rows=cross_rows,
        )
    return result
