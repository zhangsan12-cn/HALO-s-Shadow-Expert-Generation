"""Compare both selection models; all printed times are predictions.

Run: python -m examples.compare_models
The input is synthetic and its illustrative coefficients are not a calibration.
"""
import json
from pathlib import Path

from shadow_planner import (
    ShadowCountCostParams, ShadowNumaModelParams, ShadowTimeModelParams,
    evaluate_shadow_plan, plan_shadow_experts,
)


def main():
    fixture = json.loads(Path(__file__).with_name("sample_counts.json").read_text(encoding="utf-8"))
    inputs = (
        fixture["counts_mb0_by_rank"], fixture["counts_mb1_by_rank"], fixture["owner_of_expert"],
    )
    shared = dict(
        params=ShadowTimeModelParams(**fixture["time_model"]),
        cost_params=ShadowCountCostParams(**fixture["shadow_cost"]),
        numa_params=ShadowNumaModelParams(**fixture["numa"]),
    )
    output = {}
    for model in ("cost", "numa"):
        ids, metrics = plan_shadow_experts(
            *inputs, **shared, model=model, return_metrics=True, **fixture["planner"],
        )
        output[model] = {
            "shadow_ids": ids.tolist(),
            "num_selected": metrics["num_selected"],
            "predicted_own_objective_ms": metrics["objective_final"],
            "predicted_exposed_parameter_transfer_ms": metrics["param_transfer_cost_final"],
            "predicted_exposed_gradient_sync_ms": metrics["grad_sync_cost_final"],
            # Use one evaluation model when comparing two different selected sets.
            "evaluated_under_each_model": {
                evaluator: evaluate_shadow_plan(*inputs, ids, **shared, model=evaluator)
                for evaluator in ("cost", "numa")
            },
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
