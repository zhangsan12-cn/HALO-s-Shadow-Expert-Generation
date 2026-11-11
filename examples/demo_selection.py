"""Run from the repository root: python -m examples.demo_selection"""
import json
from pathlib import Path

from shadow_planner import ShadowCountCostParams, ShadowTimeModelParams, plan_shadow_experts


def main():
    fixture = json.loads(Path(__file__).with_name("sample_counts.json").read_text(encoding="utf-8"))
    ids, metrics = plan_shadow_experts(
        fixture["counts_mb0_by_rank"], fixture["counts_mb1_by_rank"],
        fixture["owner_of_expert"], model="cost",
        params=ShadowTimeModelParams(**fixture["time_model"]),
        cost_params=ShadowCountCostParams(**fixture["shadow_cost"]),
        return_metrics=True, **fixture["planner"],
    )
    print(json.dumps({
        "shadow_ids": ids.tolist(),
        "predicted_objective_ms": metrics["objective_final"],
        "exposed_parameter_transfer_ms": metrics["param_transfer_cost_final"],
        "exposed_gradient_sync_ms": metrics["grad_sync_cost_final"],
    }, indent=2))


if __name__ == "__main__":
    main()
