"""Read a portable count/configuration JSON file and emit a plan as JSON."""
import argparse
import json
from pathlib import Path

from .api import plan_shadow_experts
from .models import ShadowCountCostParams, ShadowNumaModelParams, ShadowTimeModelParams


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select shadow-expert replicas from route counts")
    parser.add_argument("input", type=Path, help="Input count/configuration JSON")
    parser.add_argument("--model", choices=("cost", "numa"), default="cost")
    parser.add_argument("--slots", type=int, help="Override M (maximum replicas per rank)")
    parser.add_argument("--output", type=Path, help="Output JSON; default: stdout")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.input.read_text(encoding="utf-8-sig"))
        options = dict(config.get("planner", {}))
        options["return_metrics"] = True
        if args.slots is not None:
            options["M"] = args.slots
        # Construct explicit defaults so CLI results do not depend on training env vars.
        time_config = dict(config["time_model"])
        if time_config.get("row_bytes") is None:
            time_config["row_bytes"] = 4096
        cost = ShadowCountCostParams(**config.get("shadow_cost", {}))
        topology_config = dict(config.get("numa", {}))
        topology_config.setdefault(
            "rank_domains", tuple(i // 2 for i in range(len(config["counts_mb0_by_rank"])))
        )
        numa = ShadowNumaModelParams(**topology_config) if args.model == "numa" else None
        ids, metrics = plan_shadow_experts(
            config["counts_mb0_by_rank"], config["counts_mb1_by_rank"],
            config["owner_of_expert"], params=ShadowTimeModelParams(**time_config),
            cost_params=cost, numa_params=numa, model=args.model, **options,
        )
        output = json.dumps(
            {"model": args.model, "shadow_ids": ids.tolist(), "metrics": metrics},
            indent=2, ensure_ascii=False, allow_nan=False,
        ) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    return 0
