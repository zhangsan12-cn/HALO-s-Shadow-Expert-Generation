"""Greedy shadow-replica selection with exact shortlist evaluation and swaps.

The algorithm predicts a two-micro-batch MoE schedule. Both topology-independent
and NUMA-aware models include exposed parameter-transfer and gradient-sync cost.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .models import (
    ShadowTimeModelParams, ShadowCountCostParams, ShadowNumaModelParams,
    _shadow_count_cost_params_from_env, _shadow_numa_params_from_env,
    _predict_shadow_T_and_gap_sum,
    shadow_additional_cost_ms, shadow_additional_cost_numa_ms,
)
from .routing import _counts_tensor_from_rows, _integer


def _plan_shadow_sets_for_layer_counts(
    counts_mb0_by_rank: Sequence[Optional[Sequence[int]]],
    counts_mb1_by_rank: Sequence[Optional[Sequence[int]]],
    owner_of_expert: Sequence[int],
    *,
    M: int = 10,
    topL: int = 64,
    tau: float = 0.5,  
    params: ShadowTimeModelParams,
    return_metrics: bool = False,
    exact_topk: int = 12,
    local_swap_rounds: int = 2,
    tau_rel: float = 0.0,
    fast_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    numa_params: Optional[ShadowNumaModelParams] = None,
    cost_params: Optional[ShadowCountCostParams] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    # Explicit parameters take precedence over compatibility environment defaults.
    count_cost_params = cost_params if cost_params is not None else _shadow_count_cost_params_from_env()
    M = _integer(M, "M")
    topL = _integer(topL, "topL")
    exact_topk = _integer(exact_topk, "exact_topk", minimum=1)
    local_swap_rounds = _integer(local_swap_rounds, "local_swap_rounds")
    tau, tau_rel = float(tau), float(tau_rel)
    if any(not math.isfinite(v) or v < 0 for v in (tau, tau_rel)):
        raise ValueError("tau and tau_rel must be finite and nonnegative")
    if len(fast_weights) != 3:
        raise ValueError("fast_weights must contain three coefficients")
    fast_weights = tuple(float(v) for v in fast_weights)
    if any(not math.isfinite(v) or v < 0 for v in fast_weights):
        raise ValueError("fast_weights must be finite and nonnegative")
    if len(counts_mb0_by_rank) != len(counts_mb1_by_rank):
        raise ValueError("counts_mb0_by_rank and counts_mb1_by_rank must have same length")
    N = len(counts_mb0_by_rank)
    if N == 0:
        raise ValueError("At least one rank is required")
    if numa_params is not None and len(numa_params.rank_domains) != N:
        raise ValueError("rank_domains must have one entry per rank")
    owner = [_integer(ow, "owner rank") for ow in owner_of_expert]
    E = len(owner)
    if E == 0:
        raise ValueError("At least one expert is required")

    if M < 0 or topL < 0 or exact_topk <= 0 or local_swap_rounds < 0:
        raise ValueError(
            f"invalid planner params: M={M}, topL={topL}, exact_topk={exact_topk}, "
            f"local_swap_rounds={local_swap_rounds}"
        )
    for e, ow in enumerate(owner):
        if ow < 0 or ow >= N:
            raise ValueError(f"owner_of_expert[{e}]={ow} out of range [0, {N})")

    C0 = _counts_tensor_from_rows(counts_mb0_by_rank, N, E)
    C1 = _counts_tensor_from_rows(counts_mb1_by_rank, N, E)
    t0 = [int(C0[i].sum().item()) for i in range(N)]
    t1 = [int(C1[i].sum().item()) for i in range(N)]
    if sum(t0) + sum(t1) >= 2**63:
        raise ValueError("Combined counts exceed the int64 range")
    demand = C0 + C1

    owned_by_rank: List[List[int]] = [[] for _ in range(N)]
    for e, ow in enumerate(owner):
        owned_by_rank[ow].append(e)

    sorted_remote: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        vals = demand[i]
        idx = torch.argsort(vals, descending=True)
        rem: List[int] = []
        for e in idx.tolist():
            if owner[e] != i and vals[e].item() > 0:
                rem.append(e)
        sorted_remote[i] = rem

    S: List[set] = [set() for _ in range(N)]
    w1, w2, w3 = float(fast_weights[0]), float(fast_weights[1]), float(fast_weights[2])
    eval_count_total = 0
    eval_count_exact = 0
    greedy_rounds = 0
    swap_accepted = 0
    num_selected = 0
    num_rejected_by_addition_cost = 0
    greedy_history = []
    swap_history = []

    def init_s_q(Cm: torch.Tensor, Tm: List[int]) -> Tuple[List[int], List[int]]:
        total_e = Cm.sum(dim=0)
        s = [0] * N
        q = [0] * N
        for i in range(N):
            owned = owned_by_rank[i]
            local_hits = int(Cm[i, owned].sum().item()) if owned else 0
            s[i] = int(Tm[i]) - local_hits
            if owned:
                q[i] = int((total_e[owned].sum() - Cm[i, owned].sum()).item())
        return s, q

    def _du(u: int, e: int) -> Tuple[int, int, int]:
        du0 = int(C0[u, e].item())
        du1 = int(C1[u, e].item())
        return du0, du1, du0 + du1

    # Keep raw token counts for expert compute; track cross-domain traffic
    # separately so topology weighting never creates fictitious compute work.
    cross_base = [[0] * N for _ in range(4)]  # send0, recv0, send1, recv1
    if numa_params is not None:
        domains = numa_params.rank_domains
        for C, offset in ((C0, 0), (C1, 2)):
            for u, row in enumerate(C.tolist()):
                for e, count in enumerate(row):
                    ow = owner[e]
                    if domains[u] != domains[ow]:
                        cross_base[offset][u] += count
                        cross_base[offset + 1][ow] += count

    def _cross_rows(changes=()):
        cross = [row[:] for row in cross_base]

        def apply(u, e, sign):
            ow = owner[e]
            if domains[u] != domains[ow]:
                du0, du1, _ = _du(u, e)
                cross[0][u] -= sign * du0
                cross[1][ow] -= sign * du0
                cross[2][u] -= sign * du1
                cross[3][ow] -= sign * du1

        for u, selected in enumerate(S):
            for e in selected:
                apply(u, e, 1)
        for u, e, sign in changes:
            apply(u, e, sign)
        return cross

    def _predict(ns0, nq0, ns1, nq1, changes=()):
        if numa_params is None:
            return _predict_shadow_T_and_gap_sum(ns0, nq0, t0, ns1, nq1, t1, params)
        cross = _cross_rows(changes)
        extra = 1.0 / numa_params.inter_bandwidth_ratio - 1.0
        communication_rows = tuple(
            [raw[i] + cross_row[i] * extra for i in range(N)]
            for raw, cross_row in zip((ns0, nq0, ns1, nq1), cross)
        )
        return _predict_shadow_T_and_gap_sum(
            ns0, nq0, t0, ns1, nq1, t1, params,
            communication_rows=communication_rows,
        )

    def _replica_counts(changes=()):
        intra = inter = 0
        for u, selected in enumerate(S):
            for e in selected:
                if domains[u] == domains[owner[e]]:
                    intra += 1
                else:
                    inter += 1
        for u, e, sign in changes:
            if domains[u] == domains[owner[e]]:
                intra += sign
            else:
                inter += sign
        return intra, inter

    def _additional_cost(changes=()):
        if numa_params is None:
            return shadow_additional_cost_ms(
                num_selected + sum(sign for _, _, sign in changes), count_cost_params
            )
        intra, inter = _replica_counts(changes)
        return shadow_additional_cost_numa_ms(intra, inter, count_cost_params, numa_params)

    s0, q0 = init_s_q(C0, t0)
    s1, q1 = init_s_q(C1, t1)
    T_curr, Gap_curr = _predict(s0, q0, s1, q1)
    T_base, Gap_base = T_curr, Gap_curr
    J_curr = T_curr

    def _tau_eff(curr_objective: float) -> float:
        # Match the acceptance objective, including exposed shadow communication.
        return max(float(tau), float(tau_rel) * float(curr_objective))

    def _fast_score(u: int, e: int) -> Optional[float]:
        ow = owner[e]
        if ow == u:
            return None
        du0, du1, du = _du(u, e)
        if du <= 0:
            return None
        su = int(s0[u] + s1[u])
        qow = int(q0[ow] + q1[ow])
        return (
            w1 * (float(du) / float(max(1, su)))
            + w2 * (float(du) / float(max(1, qow)))
            + w3 * float(du)
        )

    def _try_add(u: int, e: int) -> Optional[Tuple[float, float, List[int], List[int], List[int], List[int], float, float]]:
        ow = owner[e]
        if ow == u:
            return None
        du0, du1, du = _du(u, e)
        if du <= 0:
            return None
        ns0, nq0 = list(s0), list(q0)
        ns1, nq1 = list(s1), list(q1)
        if du0:
            ns0[u] -= du0
            nq0[ow] -= du0
        if du1:
            ns1[u] -= du1
            nq1[ow] -= du1
        newT, newGap = _predict(ns0, nq0, ns1, nq1, ((u, e, 1),))
        dT = T_curr - newT
        dGap = Gap_curr - newGap
        return dT, dGap, ns0, nq0, ns1, nq1, newT, newGap

    def _try_swap(u: int, e_out: int, e_in: int) -> Optional[Tuple[float, float, List[int], List[int], List[int], List[int], float, float]]:
        if e_out == e_in:
            return None
        ow_out = owner[e_out]
        ow_in = owner[e_in]
        if ow_in == u:
            return None
        rem0, rem1, rem = _du(u, e_out)
        add0, add1, add = _du(u, e_in)
        if rem <= 0 and add <= 0:
            return None
        ns0, nq0 = list(s0), list(q0)
        ns1, nq1 = list(s1), list(q1)
        if rem0:
            ns0[u] += rem0
            nq0[ow_out] += rem0
        if rem1:
            ns1[u] += rem1
            nq1[ow_out] += rem1
        if add0:
            ns0[u] -= add0
            nq0[ow_in] -= add0
        if add1:
            ns1[u] -= add1
            nq1[ow_in] -= add1
        newT, newGap = _predict(
            ns0, nq0, ns1, nq1, ((u, e_out, -1), (u, e_in, 1))
        )
        dT = T_curr - newT
        dGap = Gap_curr - newGap
        return dT, dGap, ns0, nq0, ns1, nq1, newT, newGap

    for _ in range(N * M if topL > 0 else 0):
        candidates: List[Tuple[int, int]] = []
        for u in range(N):
            if len(S[u]) >= M:
                continue
            cnt = 0
            for e in sorted_remote[u]:
                if e in S[u]:
                    continue
                candidates.append((u, e))
                cnt += 1
                if cnt >= topL:
                    break
        if not candidates:
            break
        greedy_rounds += 1
        eval_count_total += int(len(candidates))

        scored: List[Tuple[float, int, int]] = []
        for (u, e) in candidates:
            sc = _fast_score(u, e)
            if sc is None:
                continue
            scored.append((sc, u, e))
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        exact_candidates = [(u, e) for _, u, e in scored[: int(exact_topk)]]
        eval_count_exact += int(len(exact_candidates))

        best_primary = None
        best_tie = None
        tau_now = _tau_eff(J_curr)
        # NUMA candidates can have different transfer/sync increments.
        # Keep one shared attention window per direction for the whole set.
        cost_before = _additional_cost()
        cost_increment = (
            shadow_additional_cost_ms(num_selected + 1, count_cost_params) - cost_before
            if numa_params is None else None
        )
        evaluated = []
        for (u, e) in exact_candidates:
            res = _try_add(u, e)
            if res is None:
                continue
            dT, dGap, ns0, nq0, ns1, nq1, newT, newGap = res
            if numa_params is not None:
                cost_increment = _additional_cost(((u, e, 1),)) - cost_before
            dJ = dT - cost_increment
            old_accepts = dT > tau_now or (abs(dT) <= tau_now and dGap > 0)
            new_accepts = dJ > tau_now or (abs(dJ) <= tau_now and dGap > 0)
            # Count evaluation events, including repeated candidates in later
            # rounds; both decisions are evaluated at the SAME current state.
            if old_accepts and not new_accepts:
                num_rejected_by_addition_cost += 1
            if return_metrics:
                evaluated.append({
                    "rank": u, "expert": e, "dT_ms": dT, "dJ_ms": dJ,
                    "dGap_ms": dGap, "accepted_by_old_rule": old_accepts,
                    "accepted_by_new_rule": new_accepts,
                })
                if numa_params is not None:
                    evaluated[-1]["additional_cost_increment_ms"] = cost_increment
                    evaluated[-1]["cross_domain"] = domains[u] != domains[owner[e]]
            if dJ > tau_now:
                if (best_primary is None) or (dJ > best_primary[0] + 1e-12):
                    best_primary = (dJ, u, e, ns0, nq0, ns1, nq1, newT, newGap)
            elif abs(dJ) <= tau_now and dGap > 0:
                if (best_tie is None) or (dGap > best_tie[0] + 1e-12):
                    best_tie = (dGap, u, e, ns0, nq0, ns1, nq1, newT, newGap)

        chosen = None
        if best_primary is not None:
            _, u, e, ns0, nq0, ns1, nq1, newT, newGap = best_primary
            chosen = (u, e, ns0, nq0, ns1, nq1, newT, newGap)
        elif best_tie is not None:
            _, u, e, ns0, nq0, ns1, nq1, newT, newGap = best_tie
            chosen = (u, e, ns0, nq0, ns1, nq1, newT, newGap)
        if return_metrics:
            if numa_params is not None:
                cost_increment = (
                    _additional_cost(((chosen[0], chosen[1], 1),)) - cost_before
                    if chosen is not None else None
                )
            greedy_history.append({
                "K_before": num_selected, "T_halo_before_ms": float(T_curr),
                "objective_before_ms": float(J_curr), "tau_ms": tau_now,
                "additional_cost_increment_ms": cost_increment,
                "chosen": None if chosen is None else [chosen[0], chosen[1]],
                "chosen_branch": "primary" if best_primary is not None else (
                    "near_tie" if best_tie is not None else "stop"
                ),
                "candidates": evaluated,
            })
        if chosen is None:
            break
        u, e, s0, q0, s1, q1, T_curr, Gap_curr = chosen
        S[u].add(e)
        num_selected += 1
        J_curr = T_curr + _additional_cost()

    for _ in range(int(local_swap_rounds)):
        improved = False
        for u in range(N):
            selected = list(S[u])
            if not selected:
                continue

            pool: List[int] = []
            cnt = 0
            for e in sorted_remote[u]:
                if e in S[u]:
                    continue
                pool.append(e)
                cnt += 1
                if cnt >= topL:
                    break
            if not pool:
                continue

            pool_scored: List[Tuple[float, int]] = []
            for e in pool:
                sc = _fast_score(u, e)
                if sc is not None:
                    pool_scored.append((sc, e))
            if not pool_scored:
                continue
            pool_scored.sort(key=lambda x: x[0], reverse=True)
            pool_refine = [e for _, e in pool_scored[: int(exact_topk)]]

            best_primary = None
            best_tie = None
            tau_now = _tau_eff(J_curr)
            cost_before = _additional_cost()
            for e_out in selected:
                for e_in in pool_refine:
                    res = _try_swap(u, e_out, e_in)
                    if res is None:
                        continue
                    dT, dGap, ns0, nq0, ns1, nq1, newT, newGap = res
                    cost_increment = (
                        _additional_cost(((u, e_out, -1), (u, e_in, 1))) - cost_before
                    )
                    dJ = dT - cost_increment
                    if dJ > tau_now:
                        if (best_primary is None) or (dJ > best_primary[0] + 1e-12):
                            best_primary = (
                                dJ, e_out, e_in, ns0, nq0, ns1, nq1, newT, newGap
                            )
                    elif abs(dJ) <= tau_now and dGap > 0:
                        if (best_tie is None) or (dGap > best_tie[0] + 1e-12):
                            best_tie = (
                                dGap, e_out, e_in, ns0, nq0, ns1, nq1, newT, newGap
                            )

            chosen = None
            if best_primary is not None:
                _, e_out, e_in, ns0, nq0, ns1, nq1, newT, newGap = best_primary
                chosen = (e_out, e_in, ns0, nq0, ns1, nq1, newT, newGap)
            elif best_tie is not None:
                _, e_out, e_in, ns0, nq0, ns1, nq1, newT, newGap = best_tie
                chosen = (e_out, e_in, ns0, nq0, ns1, nq1, newT, newGap)

            if chosen is not None:
                objective_before = J_curr
                T_before = T_curr
                e_out, e_in, s0, q0, s1, q1, T_curr, Gap_curr = chosen
                S[u].discard(e_out)
                S[u].add(e_in)
                # K stays fixed, but intra/inter replica composition can change.
                J_curr = T_curr + _additional_cost()
                if return_metrics and numa_params is not None:
                    swap_history.append({
                        "rank": u, "expert_out": e_out, "expert_in": e_in,
                        "dT_ms": T_before - T_curr,
                        "additional_cost_increment_ms": _additional_cost() - cost_before,
                        "dJ_ms": objective_before - J_curr,
                    })
                swap_accepted += 1
                improved = True
        if not improved:
            break

    shadow_ids = torch.full((N, M), -1, dtype=torch.long, device="cpu")
    for i in range(N):
        ids = sorted(S[i])
        if len(ids) > M:
            ids = ids[:M]
        if ids:
            shadow_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device="cpu")

    metrics = None
    if return_metrics:
        additional_cost = _additional_cost()
        metrics = {
            "planner_variant": "numa" if numa_params is not None else "cost",
            "dJ": float(T_base - J_curr),
            "T_base": float(T_base),
            "Gap_base": float(Gap_base),
            "T_final": float(T_curr),
            "Gap_final": float(Gap_curr),
            "dT": float(T_base - T_curr),
            "dGap": float(Gap_base - Gap_curr),
            "num_greedy_rounds": int(greedy_rounds),
            "num_candidates_total": int(eval_count_total),
            "num_candidates_exact": int(eval_count_exact),
            "num_swaps_accepted": int(swap_accepted),
            "exact_topk": int(exact_topk),
            "local_swap_rounds": int(local_swap_rounds),
            "tau_abs": float(tau),
            "tau_rel": float(tau_rel),
            "fast_weights": [float(w1), float(w2), float(w3)],
            "T_halo_base": float(T_base),
            "T_halo_final": float(T_curr),
            "param_transfer_cost_final": additional_cost / 2.0,
            "grad_sync_cost_final": additional_cost / 2.0,
            "additional_cost_final": additional_cost,
            "objective_base": float(T_base),
            "objective_final": float(J_curr),
            "num_selected": int(num_selected),
            "num_selected_by_rank": [len(s) for s in S],
            "num_rejected_by_addition_cost": int(num_rejected_by_addition_cost),
            "greedy_rounds": int(greedy_rounds),
            "shadow_ids": shadow_ids.tolist(),
            "count_cost_params": None if count_cost_params is None else {
                "alpha_ms": float(count_cost_params.alpha_ms),
                "per_replica_ms": float(count_cost_params.per_replica_ms),
                "attention_ms": float(count_cost_params.attention_ms),
            },
            "greedy_history": greedy_history,
        }
        if numa_params is not None:
            intra, inter = _replica_counts()
            cross_final = _cross_rows()
            metrics.update({
                "planner_variant": "numa",
                "numa_params": {
                    "rank_domains": list(domains),
                    "inter_bandwidth_ratio": numa_params.inter_bandwidth_ratio,
                },
                "num_selected_intra": intra,
                "num_selected_inter": inter,
                "equivalent_replicas": intra + inter / numa_params.inter_bandwidth_ratio,
                "cross_domain_token_rows_base": sum(cross_base[0]) + sum(cross_base[2]),
                "cross_domain_token_rows_final": sum(cross_final[0]) + sum(cross_final[2]),
                "swap_history": swap_history,
            })
    return shadow_ids, metrics


def plan_shadow_sets_for_layer_counts_pure_compute(
    counts_mb0_by_rank: Sequence[Optional[Sequence[int]]],
    counts_mb1_by_rank: Sequence[Optional[Sequence[int]]],
    owner_of_expert: Sequence[int],
    *,
    M: int = 10,
    topL: int = 64,
    tau: float = 0.5,  # Starting threshold (ms); tune to measured time scales and noise.
    params: ShadowTimeModelParams,
    return_metrics: bool = False,
    exact_topk: int = 12,
    local_swap_rounds: int = 2,
    tau_rel: float = 0.0,
    fast_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    cost_params: Optional[ShadowCountCostParams] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Original topology-independent planner; NUMA environment variables are ignored."""
    return _plan_shadow_sets_for_layer_counts(
        counts_mb0_by_rank, counts_mb1_by_rank, owner_of_expert,
        M=M, topL=topL, tau=tau, params=params, return_metrics=return_metrics,
        exact_topk=exact_topk, local_swap_rounds=local_swap_rounds,
        tau_rel=tau_rel, fast_weights=fast_weights, cost_params=cost_params,
    )


def plan_shadow_sets_for_layer_counts_numa(
    counts_mb0_by_rank: Sequence[Optional[Sequence[int]]],
    counts_mb1_by_rank: Sequence[Optional[Sequence[int]]],
    owner_of_expert: Sequence[int],
    *,
    M: int = 10,
    topL: int = 64,
    tau: float = 0.5,  # Starting threshold (ms); tune to measured time scales and noise.
    params: ShadowTimeModelParams,
    return_metrics: bool = False,
    exact_topk: int = 12,
    local_swap_rounds: int = 2,
    tau_rel: float = 0.0,
    fast_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    numa_params: Optional[ShadowNumaModelParams] = None,
    cost_params: Optional[ShadowCountCostParams] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """NUMA-aware counterpart with the same candidate shortlist and compute schedule.

    Only token communication and replica transfer/sync costs are topology weighted.
    The shortlist remains unchanged for comparison with the original method.
    With no explicit topology, read MOE_SHADOW_NUMA_RANK_DOMAINS and
    MOE_SHADOW_NUMA_INTER_BANDWIDTH_RATIO (defaults: adjacent pairs and 0.79).
    A ratio of 1 recovers the original objective and selection decisions.
    Like the original model, this does not model link contention, dedup traffic,
    disabled-local-compute scheduling, or communication-frequency amortization.
    """
    if numa_params is None:
        numa_params = _shadow_numa_params_from_env(len(counts_mb0_by_rank))
    return _plan_shadow_sets_for_layer_counts(
        counts_mb0_by_rank, counts_mb1_by_rank, owner_of_expert,
        M=M, topL=topL, tau=tau, params=params, return_metrics=return_metrics,
        exact_topk=exact_topk, local_swap_rounds=local_swap_rounds,
        tau_rel=tau_rel, fast_weights=fast_weights, cost_params=cost_params,
        numa_params=numa_params,
    )
