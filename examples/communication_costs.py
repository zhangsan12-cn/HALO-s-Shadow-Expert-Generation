"""Show exposed costs with the 7.636 ms starting attention window.

Run: python -m examples.communication_costs
NUMA case below deliberately places every replica across a domain boundary.
Replace the example coefficients/window with measured values for real experiments.
"""
import json

from shadow_planner import (
    ShadowCountCostParams, ShadowNumaModelParams,
    shadow_additional_cost_ms, shadow_additional_cost_numa_ms,
)


def main():
    cost = ShadowCountCostParams()
    topology = ShadowNumaModelParams()
    output = []
    for replicas in (0, 4, 8, 16, 24, 32, 40):
        homogeneous = shadow_additional_cost_ms(replicas, cost)
        numa = shadow_additional_cost_numa_ms(0, replicas, cost, topology)
        output.append({
            "replicas": replicas,
            "cost_model_exposed_total_ms": homogeneous,
            "numa_all_cross_domain_exposed_total_ms": numa,
            "numa_parameter_transfer_ms": numa / 2,
            "numa_gradient_sync_ms": numa / 2,
        })
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
