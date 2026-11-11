import json
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import torch

from shadow_planner import (
    ShadowCountCostParams, ShadowNumaModelParams, ShadowTimeModelParams,
    evaluate_shadow_plan, plan_shadow_experts,
    plan_shadow_sets_for_layer_counts_numa,
    plan_shadow_sets_for_layer_counts_pure_compute,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    config = json.loads((ROOT / "examples" / "sample_counts.json").read_text(encoding="utf-8"))
    inputs = (config["counts_mb0_by_rank"], config["counts_mb1_by_rank"], config["owner_of_expert"])
    # Preserve the explicit historical regression scenario, independently of the
    # package/demo defaults. Its zero window exposes path-specific swap costs.
    cost_config = dict(config["shadow_cost"], attention_ms=0.0)
    search_config = dict(config["planner"], tau=0.0001)
    shared = dict(
        params=ShadowTimeModelParams(**config["time_model"]),
        cost_params=ShadowCountCostParams(**cost_config),
        return_metrics=True, **search_config,
    )
    return inputs, shared


class SelectionTests(unittest.TestCase):
    def test_known_reference_outputs_for_both_models(self):
        inputs, shared = fixture()
        expected = {
            "cost": [[-1, -1], [-1, -1], [2, 11], [3, -1]],
            "numa": [[-1, -1], [9, -1], [2, 3], [3, -1]],
        }
        for model in expected:
            ids, metrics = plan_shadow_experts(*inputs, **shared, model=model, numa_params=ShadowNumaModelParams())
            self.assertEqual(ids.tolist(), expected[model])
            self.assertEqual(metrics["planner_variant"], model)
            self.assertEqual(ids.device.type, "cpu")
            self.assertEqual(ids.dtype, torch.int64)

    def test_legacy_entrypoints_use_the_same_algorithm(self):
        inputs, shared = fixture()
        for model, function in (
            ("cost", plan_shadow_sets_for_layer_counts_pure_compute),
            ("numa", plan_shadow_sets_for_layer_counts_numa),
        ):
            extra = {"numa_params": ShadowNumaModelParams()} if model == "numa" else {}
            ids, metrics = plan_shadow_experts(*inputs, **shared, model=model, **extra)
            legacy_ids, legacy_metrics = function(*inputs, **shared, **extra)
            self.assertTrue(torch.equal(ids, legacy_ids))
            self.assertEqual(metrics, legacy_metrics)

    def test_ratio_one_recovers_original_choices_and_objective(self):
        for seed in range(20):
            rng = random.Random(seed)
            owner = [e // 2 for e in range(8)]
            counts = [[[rng.randrange(3000) for _ in owner] for _ in range(4)] for _ in range(2)]
            for attention in (0, 7.636):
                options = dict(
                    params=ShadowTimeModelParams(0.2034, 0.0889, 0.8, 0.046500382, row_bytes=4096),
                    cost_params=ShadowCountCostParams(attention_ms=attention),
                    M=seed % 4, exact_topk=1 + seed % 12, return_metrics=True,
                )
                ids, metrics = plan_shadow_experts(*counts, owner, **options)
                numa_ids, numa_metrics = plan_shadow_experts(
                    *counts, owner, **options, model="numa",
                    numa_params=ShadowNumaModelParams(inter_bandwidth_ratio=1),
                )
                self.assertTrue(torch.equal(ids, numa_ids))
                for key in ("T_base", "T_final", "objective_final", "additional_cost_final", "num_swaps_accepted"):
                    self.assertEqual(metrics[key], numa_metrics[key], (seed, attention, key))

    def test_all_ranks_in_one_domain_recover_original(self):
        inputs, shared = fixture()
        ids, metrics = plan_shadow_experts(*inputs, **shared)
        numa_ids, numa_metrics = plan_shadow_experts(
            *inputs, **shared, model="numa", numa_params=ShadowNumaModelParams((0, 0, 0, 0), 0.79),
        )
        self.assertTrue(torch.equal(ids, numa_ids))
        self.assertEqual(metrics["objective_final"], numa_metrics["objective_final"])

    def test_rebuilding_returned_plan_recovers_final_objective(self):
        inputs, shared = fixture()
        for model in ("cost", "numa"):
            ids, metrics = plan_shadow_experts(*inputs, **shared, model=model, numa_params=ShadowNumaModelParams())
            score = evaluate_shadow_plan(
                *inputs, ids, params=shared["params"], cost_params=shared["cost_params"],
                model=model, numa_params=ShadowNumaModelParams(),
            )
            self.assertAlmostEqual(score["objective"], metrics["objective_final"])
            self.assertAlmostEqual(score["T_halo"], metrics["T_final"])
            self.assertAlmostEqual(score["param_transfer_cost"], metrics["param_transfer_cost_final"])
            self.assertAlmostEqual(score["grad_sync_cost"], metrics["grad_sync_cost_final"])
            self.assertEqual(score["num_selected"], metrics["num_selected"])
            if model == "numa":
                self.assertEqual(score["cross_domain_token_rows"], metrics["cross_domain_token_rows_final"])

    def test_swap_uses_domain_specific_cost_change(self):
        inputs, shared = fixture()
        _, metrics = plan_shadow_experts(*inputs, **shared, model="numa", numa_params=ShadowNumaModelParams())
        changes = [s for s in metrics["swap_history"] if abs(s["additional_cost_increment_ms"]) > 1e-9]
        self.assertTrue(changes)
        for swap in changes:
            self.assertAlmostEqual(swap["dJ_ms"], swap["dT_ms"] - swap["additional_cost_increment_ms"])
            self.assertGreater(swap["dJ_ms"], shared["tau"])

    def test_candidates_do_not_share_one_cost_increment(self):
        counts = [[0 if i == e else 10 for e in range(4)] for i in range(4)]
        _, metrics = plan_shadow_experts(
            counts, counts, list(range(4)), model="numa", numa_params=ShadowNumaModelParams(),
            params=ShadowTimeModelParams(0, 1, 0, 0, row_bytes=1, mib_bytes=1),
            cost_params=ShadowCountCostParams(attention_ms=0), M=1,
            exact_topk=12, return_metrics=True,
        )
        candidates = metrics["greedy_history"][0]["candidates"]
        self.assertEqual({c["cross_domain"] for c in candidates}, {False, True})
        self.assertEqual(len({c["additional_cost_increment_ms"] for c in candidates}), 2)

    def test_explicit_parameters_override_environment(self):
        inputs, shared = fixture()
        ids, metrics = plan_shadow_experts(*inputs, **shared, model="numa", numa_params=ShadowNumaModelParams())
        with patch.dict(os.environ, {
            "MOE_SHADOW_COUNT_COST_ALPHA_MS": "invalid",
            "MOE_SHADOW_COUNT_COST_ATTENTION_MS": "invalid",
            "MOE_SHADOW_NUMA_RANK_DOMAINS": "invalid",
            "MOE_SHADOW_NUMA_INTER_BANDWIDTH_RATIO": "invalid",
        }):
            other_ids, other = plan_shadow_experts(*inputs, **shared, model="numa", numa_params=ShadowNumaModelParams())
        self.assertTrue(torch.equal(ids, other_ids))
        self.assertEqual(metrics, other)

    def test_relative_tolerance_uses_total_objective_for_additions(self):
        # After the first replica: MoE time=2, exposed cost=1.6, total=3.6.
        # The next replica saves 0.4 with no gap improvement. At tau_rel=0.15,
        # it exceeds the MoE-only tolerance 0.3 but not the total tolerance 0.54.
        counts0 = [[0, 10], [1, 0]]
        counts1 = [[0, 10], [0, 0]]
        params = ShadowTimeModelParams(0, 1, 0, 0, row_bytes=1, mib_bytes=1)
        costs = ShadowCountCostParams(alpha_ms=0, per_replica_ms=0.8, attention_ms=0)
        for model in ("cost", "numa"):
            scale = 1 if model == "cost" else 1 / 0.79
            for relative in (0, 0.15):
                with self.subTest(model=model, tau_rel=relative):
                    ids, metrics = plan_shadow_experts(
                        counts0, counts1, [0, 1], params=params, cost_params=costs,
                        model=model, numa_params=ShadowNumaModelParams((0, 1), 0.79),
                        M=1, topL=1, exact_topk=2, local_swap_rounds=0,
                        tau=0.1, tau_rel=relative, return_metrics=True,
                    )
                    expected = [[1], [0]] if relative == 0 else [[1], [-1]]
                    self.assertEqual(ids.tolist(), expected)
                    second = metrics["greedy_history"][1]
                    self.assertAlmostEqual(second["T_halo_before_ms"], 2 * scale)
                    self.assertAlmostEqual(second["objective_before_ms"], 3.6 * scale)
                    self.assertAlmostEqual(second["tau_ms"], max(0.1, relative * 3.6 * scale))
                    candidate = second["candidates"][0]
                    self.assertAlmostEqual(candidate["dJ_ms"], 0.4 * scale)
                    self.assertEqual(candidate["dGap_ms"], 0)
                    if relative:
                        self.assertGreater(candidate["dJ_ms"], relative * 2 * scale)
                        self.assertLess(candidate["dJ_ms"], second["tau_ms"])
                        self.assertEqual(second["chosen_branch"], "stop")
                    else:
                        self.assertEqual(second["chosen_branch"], "primary")

    def test_relative_tolerance_uses_total_objective_for_swaps(self):
        # At tau_rel=0.05, greedy selection has MoE time=148 and exposed cost=18. Swapping
        # rank 2's expert 1 for expert 2 saves 8 without improving the gap.
        # It exceeds 0.05*148=7.4, but not the total tolerance 0.05*166=8.3.
        counts0 = [
            [8, 3, 7, 2, 2, 8],
            [8, 5, 13, 5, 16, 3],
            [10, 8, 11, 11, 20, 2],
        ]
        counts1 = [
            [7, 8, 14, 14, 20, 5],
            [19, 13, 12, 19, 2, 8],
            [2, 10, 5, 0, 14, 18],
        ]
        owner = [0, 0, 1, 1, 2, 2]
        params = ShadowTimeModelParams(0, 1, 0, 1, row_bytes=1, mib_bytes=1)
        costs = ShadowCountCostParams(alpha_ms=0, per_replica_ms=3, attention_ms=0)
        for model in ("cost", "numa"):
            options = dict(
                params=params, cost_params=costs, model=model,
                numa_params=ShadowNumaModelParams((0, 0, 0), 1),
            )
            before = evaluate_shadow_plan(counts0, counts1, owner, [[4], [0], [1]], **options)
            after = evaluate_shadow_plan(counts0, counts1, owner, [[4], [0], [2]], **options)
            self.assertAlmostEqual(before["T_halo"], 148)
            self.assertAlmostEqual(before["additional_cost"], 18)
            self.assertAlmostEqual(before["objective"] - after["objective"], 8)
            self.assertEqual(before["gap"], after["gap"])
            for relative in (0, 0.05):
                with self.subTest(model=model, tau_rel=relative):
                    ids, metrics = plan_shadow_experts(
                        counts0, counts1, owner, **options,
                        M=1, topL=4, exact_topk=1, local_swap_rounds=2,
                        tau=0.01, tau_rel=relative, return_metrics=True,
                    )
                    expected = [[-1], [-1], [-1]] if relative == 0 else [[4], [0], [1]]
                    self.assertEqual(ids.tolist(), expected)
                    self.assertEqual(metrics["num_swaps_accepted"], 0)
                    self.assertAlmostEqual(metrics["objective_final"], 190 if relative == 0 else 166)


    def test_python_defaults_and_environment_fallback_are_consistent(self):
        import inspect

        for function in (
            plan_shadow_experts, plan_shadow_sets_for_layer_counts_pure_compute,
            plan_shadow_sets_for_layer_counts_numa,
        ):
            self.assertEqual(inspect.signature(function).parameters["tau"].default, 0.5)
        inputs, shared = fixture()
        with patch.dict(os.environ, {}, clear=True):
            _, metrics = plan_shadow_experts(
                *inputs, params=shared["params"], M=0, return_metrics=True,
            )
        self.assertEqual(metrics["tau_abs"], 0.5)
        self.assertEqual(metrics["count_cost_params"]["attention_ms"], 7.636)

    def test_zero_slots_zero_shortlist_and_missing_rows(self):
        inputs, shared = fixture()
        for option in ({"M": 0}, {"topL": 0}):
            args = dict(shared, **option)
            ids, metrics = plan_shadow_experts(*inputs, **args)
            self.assertTrue((ids == -1).all().item())
            self.assertEqual(metrics["num_selected"], 0)
        zeros = [[0] * 4 for _ in range(4)]
        ids, metrics = plan_shadow_experts(
            [None] * 4, zeros, list(range(4)), params=shared["params"],
            cost_params=shared["cost_params"], M=1,
        )
        self.assertIsNone(metrics)
        self.assertTrue((ids == -1).all().item())

    def test_invalid_inputs_fail_early(self):
        inputs, shared = fixture()
        for overrides in ({"M": -1}, {"M": 1.5}, {"exact_topk": 0}, {"tau": -1}, {"fast_weights": (1, 2)}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                plan_shadow_experts(*inputs, **dict(shared, **overrides))
        with self.assertRaises(ValueError):
            plan_shadow_experts(*inputs, **shared, model="unknown")
        with self.assertRaises(ValueError):
            plan_shadow_experts(*inputs, **shared, model="numa", numa_params=ShadowNumaModelParams((0, 1)))
        for bad in (-1, 1.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_shadow_experts([[bad]], [[0]], [0], params=shared["params"])

    def test_evaluator_rejects_invalid_plans(self):
        inputs, shared = fixture()
        invalid = (
            [[0], [-1], [-1], [-1]],  # Expert 0 is already owned by rank 0.
            [[3, 3], [-1, -1], [-1, -1], [-1, -1]],
            [[99], [-1], [-1], [-1]],
        )
        for ids in invalid:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                evaluate_shadow_plan(*inputs, ids, params=shared["params"], cost_params=shared["cost_params"])


if __name__ == "__main__":
    unittest.main()
