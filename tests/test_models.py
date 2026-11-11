import math
import unittest

from shadow_planner import (
    ShadowCountCostParams, ShadowNumaModelParams, ShadowTimeModelParams,
    default_owner_map_for_shadow, evaluate_shadow_plan,
    shadow_additional_cost_ms, shadow_additional_cost_numa_ms,
)


class TimeModelTests(unittest.TestCase):
    def test_empty_set_has_no_shadow_cost_even_with_startup(self):
        costs = ShadowCountCostParams(alpha_ms=10, per_replica_ms=1, attention_ms=0)
        self.assertEqual(shadow_additional_cost_ms(0, costs), 0)
        self.assertEqual(shadow_additional_cost_numa_ms(0, 0, costs, ShadowNumaModelParams()), 0)

    def test_attention_hides_only_the_overlapped_part(self):
        costs = ShadowCountCostParams(alpha_ms=0.2, per_replica_ms=1, attention_ms=2)
        self.assertEqual(shadow_additional_cost_ms(1, costs), 0)
        self.assertAlmostEqual(shadow_additional_cost_ms(3, costs), 2.4)

    def test_numa_replica_cost_keeps_one_window_per_direction(self):
        costs = ShadowCountCostParams(alpha_ms=0.2, per_replica_ms=1, attention_ms=2)
        topology = ShadowNumaModelParams(inter_bandwidth_ratio=0.5)
        # Equivalent replicas = 1 + 1/0.5 = 3. No per-replica window subtraction.
        self.assertAlmostEqual(shadow_additional_cost_numa_ms(1, 1, costs, topology), 2.4)

    def test_cross_domain_slows_only_communication(self):
        zero = [[0] * 4 for _ in range(4)]
        params = ShadowTimeModelParams(0, 1, 0, 2, row_bytes=1, mib_bytes=1)
        plan = [[] for _ in range(4)]
        for destination, expected in ((1, 40), (2, 20 + 20 / 0.79)):
            counts = [[0] * 4 for _ in range(4)]
            counts[0][destination] = 10
            score = evaluate_shadow_plan(
                counts, zero, list(range(4)), plan,
                params=params, model="numa", numa_params=ShadowNumaModelParams(),
                cost_params=ShadowCountCostParams(),
            )
            # Dispatch 10/r + remote compute 20 + gather 10/r.
            self.assertAlmostEqual(score["T_halo"], expected)

    def test_default_window_and_exposed_cost(self):
        self.assertEqual(ShadowCountCostParams().attention_ms, 7.636)
        self.assertAlmostEqual(shadow_additional_cost_ms(8, ShadowCountCostParams()), 8.6044)

    def test_parameter_validation(self):
        for value in (-1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ShadowCountCostParams(per_replica_ms=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                ShadowTimeModelParams(value, 1, 0, 1)
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ShadowNumaModelParams(inter_bandwidth_ratio=value)
        with self.assertRaises(ValueError):
            ShadowTimeModelParams(0, 1, 0, 1, row_bytes=0)
        with self.assertRaises(ValueError):
            ShadowTimeModelParams(0, 1, 0, 1, assume_g_equals_d=False)
        with self.assertRaises(ValueError):
            shadow_additional_cost_ms(1.5, ShadowCountCostParams())

    def test_owner_mapping(self):
        self.assertEqual(default_owner_map_for_shadow(4, 8), [0, 0, 1, 1, 2, 2, 3, 3])
        with self.assertRaises(ValueError):
            default_owner_map_for_shadow(4, 7)


if __name__ == "__main__":
    unittest.main()
