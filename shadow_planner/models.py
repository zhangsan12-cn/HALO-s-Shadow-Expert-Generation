"""Time models for MoE execution and exposed shadow-replica communication.

All times are milliseconds. Bandwidth terms use MiB, not decimal megabytes.
No distributed communication, model weights, CUDA, or training hooks are used.
"""
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .routing import _integer


@dataclass(frozen=True)
class ShadowTimeModelParams:
    """MoE linear timing coefficients (ms and ms/MiB).

    row_bytes is bytes per token/expert route row, not bytes per expert.
    Gather time is approximated by dispatch time in the current schedule.
    Fit alpha/beta/gamma/delta to measurements of the actual hardware and workload;
    derive row_bytes from the actual hidden size and payload dtype.
    """

    alpha: float
    beta: float
    gamma: float
    delta: float
    assume_g_equals_d: bool = True
    # Unit conversion for counts -> MiB in shadow time model:
    # x_mib = x_rows * row_bytes / mib_bytes
    # Typical bf16 default: row_bytes = hidden_size * 2 (bytes).
    row_bytes: Optional[float] = None
    mib_bytes: float = float(1024 * 1024)

    def __post_init__(self):
        for name in ("alpha", "beta", "gamma", "delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("row_bytes", "mib_bytes"):
            value = getattr(self, name)
            if value is None and name == "row_bytes":
                continue
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not self.assume_g_equals_d:
            raise ValueError("Only assume_g_equals_d=True is supported by this timing schedule")


@dataclass(frozen=True)
class ShadowCountCostParams:
    """Count-only, symmetric exposed communication cost, in milliseconds.

    K counts (rank, expert) replicas globally, not distinct logical experts or M.
    One forward transfer and one backward sync are charged, without frequency
    amortization. The global linear traffic approximation ignores concurrency.
    Public planners always include this cost; coefficients may be set to zero.
    """

    # Example coefficients only: refit for the actual links and expert payload size.
    alpha_ms: float = 0.2034
    per_replica_ms: float = 1.46685
    # Starting overlap window (ms): replace with the measured usable attention window.
    # The shared forward/backward window is a modeling approximation, not a constant.
    attention_ms: float = 7.636

    def __post_init__(self):
        for name in ("alpha_ms", "per_replica_ms", "attention_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value}")
            object.__setattr__(self, name, value)


def _shadow_count_cost_params_from_env() -> ShadowCountCostParams:
    """Compatibility fallback when no explicit ShadowCountCostParams is supplied."""
    return ShadowCountCostParams(
        alpha_ms=float(os.environ.get("MOE_SHADOW_COUNT_COST_ALPHA_MS", "0.2034")),
        per_replica_ms=float(
            os.environ.get("MOE_SHADOW_COUNT_COST_PER_REPLICA_MS", "1.46685")
        ),
        attention_ms=float(
            # Same starting window as the dataclass; override with measured values.
            os.environ.get("MOE_SHADOW_COUNT_COST_ATTENTION_MS", "7.636")
        ),
    )


def shadow_additional_cost_ms(
    num_selected: int, cost_params: Optional[ShadowCountCostParams]
) -> float:
    """Both directions: 2*max(0, alpha+c*K-A), with zero cost for K=0."""
    num_selected = _integer(num_selected, "num_selected")
    if cost_params is None or num_selected == 0:
        return 0.0
    return 2.0 * max(
        0.0,
        float(cost_params.alpha_ms)
        + float(cost_params.per_replica_ms) * num_selected
        - float(cost_params.attention_ms),
    )


@dataclass(frozen=True)
class ShadowNumaModelParams:
    """Rank-indexed topology; inter-domain bandwidth / intra-domain bandwidth.

    Defaults describe ranks 0/1 in one domain and ranks 2/3 in another.
    Existing communication coefficients are interpreted as intra-domain values.
    Only the bandwidth-dependent term is scaled; startup latency is unchanged.
    This is an additive traffic approximation, not a physical-link simulator.
    Replace the example rank mapping and ratio with the actual topology and
    measured inter-/intra-domain bandwidth ratio for the relevant payload sizes.
    """

    rank_domains: Tuple[int, ...] = (0, 0, 1, 1)
    inter_bandwidth_ratio: float = 0.79

    def __post_init__(self):
        domains = tuple(_integer(d, "domain id") for d in self.rank_domains)
        if not domains:
            raise ValueError("rank_domains must not be empty")
        ratio = float(self.inter_bandwidth_ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("inter_bandwidth_ratio must be finite and positive")
        object.__setattr__(self, "rank_domains", domains)
        object.__setattr__(self, "inter_bandwidth_ratio", ratio)


def _shadow_numa_params_from_env(num_ranks: int) -> ShadowNumaModelParams:
    # Rank ids follow the training process mapping, not physical GPU ids.
    # With no override, adjacent rank pairs share a domain (0,0,1,1 for 4 ranks).
    mapping = os.environ.get("MOE_SHADOW_NUMA_RANK_DOMAINS")
    domains = (
        tuple(int(value.strip()) for value in mapping.split(","))
        if mapping is not None
        else tuple(rank // 2 for rank in range(num_ranks))
    )
    if len(domains) != num_ranks:
        raise ValueError("MOE_SHADOW_NUMA_RANK_DOMAINS must have one entry per rank")
    return ShadowNumaModelParams(
        rank_domains=domains,
        inter_bandwidth_ratio=float(
            os.environ.get("MOE_SHADOW_NUMA_INTER_BANDWIDTH_RATIO", "0.79")
        ),
    )


def shadow_additional_cost_numa_ms(
    num_intra: int,
    num_inter: int,
    cost_params: Optional[ShadowCountCostParams],
    numa_params: ShadowNumaModelParams,
) -> float:
    """Exposed parameter-transfer + gradient-sync cost for a replica set.

    Charge one shared attention window per direction, as in the original model.
    Inter-domain replicas consume 1 / bandwidth_ratio equivalent replicas.
    Forward/backward symmetry and the original frequency assumptions are kept.
    """
    num_intra = _integer(num_intra, "num_intra")
    num_inter = _integer(num_inter, "num_inter")
    if cost_params is None or num_intra + num_inter == 0:
        return 0.0
    equivalent_replicas = num_intra + num_inter / numa_params.inter_bandwidth_ratio
    return 2.0 * max(
        0.0,
        cost_params.alpha_ms
        + cost_params.per_replica_ms * equivalent_replicas
        - cost_params.attention_ms,
    )


def default_owner_map_for_shadow(num_ranks: int, num_experts: int) -> List[int]:
    num_ranks = _integer(num_ranks, "num_ranks", minimum=1)
    num_experts = _integer(num_experts, "num_experts", minimum=1)
    if num_experts % num_ranks != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
        )
    experts_per_rank = num_experts // num_ranks
    return [e // experts_per_rank for e in range(num_experts)]


def _predict_shadow_T_and_gap_sum(
    s0: List[int],
    q0: List[int],
    t0: List[int],
    s1: List[int],
    q1: List[int],
    t1: List[int],
    params: ShadowTimeModelParams,
    *,
    communication_rows: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]] = None,
) -> Tuple[float, float]:
    n_ranks = len(s0)
    # Convert routing counts (rows = token*top_k entries) into MiB before
    # applying linear coefficients beta/delta. This keeps beta/delta in ms/MiB.
    row_bytes = params.row_bytes
    if row_bytes is None:
        row_bytes = float(os.environ.get("MOE_SHADOW_ROW_BYTES", "4096"))
    row_bytes = float(row_bytes)
    if not math.isfinite(row_bytes) or row_bytes <= 0:
        raise ValueError("row_bytes must be finite and positive")
    mib_bytes = float(params.mib_bytes)

    def rows_to_mib(x_rows: int) -> float:
        return float(x_rows) * row_bytes / mib_bytes

    def d_from_sq(s: List[int], q: List[int]) -> float:
        mx = 0
        for i in range(n_ranks):
            v = s[i] if s[i] > q[i] else q[i]
            if v > mx:
                mx = v
        return params.alpha + params.beta * rows_to_mib(mx)

    def l_from_h(h: int) -> float:
        return params.gamma + params.delta * rows_to_mib(h)

    def r_from_q(qv: int) -> float:
        return params.gamma + params.delta * rows_to_mib(qv)

    h0 = [t0[i] - s0[i] for i in range(n_ranks)]
    h1 = [t1[i] - s1[i] for i in range(n_ranks)]
    l0 = [l_from_h(h0[i]) for i in range(n_ranks)]
    l1 = [l_from_h(h1[i]) for i in range(n_ranks)]
    r0 = [r_from_q(q0[i]) for i in range(n_ranks)]
    r1 = [r_from_q(q1[i]) for i in range(n_ranks)]

    # Weighted rows affect communication only; h=t-s and remote compute q
    # must continue to use actual token rows.
    if communication_rows is None:
        d0 = d_from_sq(s0, q0)
        d1 = d_from_sq(s1, q1)
    else:
        cs0, cq0, cs1, cq1 = communication_rows
        d0 = d_from_sq(cs0, cq0)
        d1 = d_from_sq(cs1, cq1)
    g0, g1 = d0, d1

    tD0 = d0
    tL0 = l0[:]
    tD1 = tD0 + d1
    tR0 = [max(tL0[i], tD0) + r0[i] for i in range(n_ranks)]
    tR1 = [max(tR0[i], tD1) + r1[i] for i in range(n_ranks)]
    tG0 = max(tD1, max(tR0)) + g0
    tG1 = max(tG0, max(tR1)) + g1
    tL1 = [tR1[i] + l1[i] for i in range(n_ranks)]
    T = max(tG1, max(tL1))

    gap_sum = 0.0
    for i in range(n_ranks):
        gi = g1 - l1[i]
        if gi > 0.0:
            gap_sum += gi
    return T, gap_sum
