#!/usr/bin/env python3
"""Bounded, synthetic gates for the run-local keyed scorer."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_results import (  # noqa: E402
    _bootstrap_report,
    index_rows,
    prefix_reachability_metrics,
    score_arm_rows,
)


class Top100CommonSupportScorerTest(unittest.TestCase):
    def test_single_model_bootstrap_with_two_cases_per_axis(self):
        axes = ['axis_a', 'axis_b']
        conditions = ['reference', 'true', 'shuffled']
        frozen = [{'group_uid': f'{axis}_{case}_{sign}', 'case_id': f'{axis}_{case}',
                   'axis_id': axis, 'polarity': sign}
                  for axis in axes for case in range(2) for sign in (-1, 1)]
        scores = {f'top30.{arm}': [dict(row, metrics={'value': offset}) for row in frozen]
                  for arm, offset in [('reference', 0.1), ('true', 0.5), ('shuffled', 0.2)]}
        point, bootstrap = _bootstrap_report(
            frozen_rows=frozen, scored_by_arm=scores, models=['top30'],
            conditions=conditions, axes=axes, all_metric_names=['value'],
            bootstrap_metric_names=['value'], replicates=20, seed=1042, batch_size=8)
        self.assertEqual(point['group_micro']['top30.true']['value'], 0.5)
        self.assertEqual(bootstrap['status'], 'passed')
        estimands = bootstrap['overall_case_cluster']
        self.assertFalse(any('top30_minus_top30' in name for name in estimands))
        self.assertAlmostEqual(estimands['top30.true_minus_shuffled']['observed']['value'], 0.3)

    def test_literal_prefix_reachability(self) -> None:
        metrics = prefix_reachability_metrics(
            ["C", "C", "CC", None, "O"],
            top100_endpoints={"C", "CC"},
            positive_endpoints={"C"},
            negative_endpoints={"CC"},
            budgets=(1, 5),
        )
        self.assertEqual(metrics["budget_1_unique_top100_hits"], 1.0)
        self.assertEqual(metrics["budget_1_top100_recall"], 0.5)
        self.assertAlmostEqual(metrics["budget_5_validity"], 0.8)
        self.assertEqual(metrics["budget_5_unique_valid_endpoints"], 3.0)
        self.assertAlmostEqual(metrics["budget_5_duplicate_rate"], 0.2)
        self.assertEqual(metrics["budget_5_unique_top100_hits"], 2.0)
        self.assertEqual(metrics["budget_5_top100_recall"], 1.0)
        self.assertAlmostEqual(metrics["budget_5_valid_in_top100_rate"], 0.6)
        self.assertAlmostEqual(metrics["budget_5_valid_outside_top100_rate"], 0.2)

    @staticmethod
    def _fixtures() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        frozen: list[dict[str, object]] = []
        generated: list[dict[str, object]] = []
        for index, case_id in enumerate(("case_a", "case_b")):
            uid = f"group_{index}"
            identity = {
                "group_uid": uid,
                "case_id": case_id,
                "evaluation_group_index": index,
                "legacy_group_idx": index,
                "root_draw_seeds": [100 + index * 3 + value for value in range(3)],
                "root_choice_indices": [0, 1, 2],
                "trajectory_seeds": [200 + index * 3 + value for value in range(3)],
            }
            frozen.append(
                {
                    **identity,
                    "axis_id": "rdkit.ring_count",
                    "polarity": 1 if index == 0 else -1,
                    "positive_target_smiles": ["C"],
                    "negative_target_smiles": ["CC"],
                }
            )
            generated.append(
                {
                    **identity,
                    "arm": "true",
                    "guidance_residual_scale": 0.5,
                    "execution_profile_id": "synthetic_profile",
                    "sampling_steps": 50,
                    "per_sample_transition_kernel": "stateless_counter_v1",
                    "sampling_rng_mode": "paired_stateless_counter_v1",
                    "generated_smiles": ["C", "CC", "O"],
                    "categories": ["positive", "negative", "other_valid"],
                    "rc_permutation_keys": ["a", "b", "c"],
                }
            )
        return frozen, generated

    def test_keyed_scoring_is_invariant_to_generated_row_order(self) -> None:
        frozen, generated = self._fixtures()
        inventories = {"case_a": {"C", "CC", "O"}, "case_b": {"C", "CC", "O"}}
        forward, forward_pool = score_arm_rows(
            frozen_rows=frozen,
            inventory_by_case=inventories,
            generated_rows=generated,
            model="synthetic",
            condition="true",
            alpha=0.5,
            draw_count=3,
            sampling_steps=50,
            profile_id="synthetic_profile",
            top_k=(1,),
            budgets=(1, 3),
        )
        reverse, reverse_pool = score_arm_rows(
            frozen_rows=frozen,
            inventory_by_case=inventories,
            generated_rows=list(reversed(generated)),
            model="synthetic",
            condition="true",
            alpha=0.5,
            draw_count=3,
            sampling_steps=50,
            profile_id="synthetic_profile",
            top_k=(1,),
            budgets=(1, 3),
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(forward_pool, reverse_pool)
        self.assertEqual([row["group_uid"] for row in forward], ["group_0", "group_1"])

    def test_duplicate_and_missing_group_keys_fail_closed(self) -> None:
        frozen, generated = self._fixtures()
        with self.assertRaisesRegex(RuntimeError, "repeats group_uid"):
            index_rows([generated[0], copy.deepcopy(generated[0])], key="group_uid", label="test")
        with self.assertRaisesRegex(RuntimeError, "coverage differs"):
            score_arm_rows(
                frozen_rows=frozen,
                inventory_by_case={"case_a": {"C", "CC"}, "case_b": {"C", "CC"}},
                generated_rows=generated[:1],
                model="synthetic",
                condition="true",
                alpha=0.5,
                draw_count=3,
                sampling_steps=50,
                profile_id="synthetic_profile",
                top_k=(1,),
                budgets=(1, 3),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
