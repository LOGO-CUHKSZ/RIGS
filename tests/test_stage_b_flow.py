"""Bounded CPU integration checks with synthetic states and sampler outputs.

The selection, tensor export/load, and scoring are real. GPU sampling is
replaced by deterministic chemistry fixtures; this is not a retraining run.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce
import score_results
import stage_b_flow as flow
from retro_guided_generation.evaluation.pista import (
    PISTA_FROZEN_SCHEMA, _load_pista_model,
    root_choice_indices_for_group, root_draw_seeds_for_group, trajectory_seeds_for_group,
)


class StageBFlowTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.args = SimpleNamespace(run_dir=self.run, device='cuda:0', shard_count=1, shard_index=0)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.calls = []
        write = reproduce.write_json
        root = self.run
        write(root/'prepared.json', {'data_root': str(root/'data_package'),
                                    'checkpoint': 'released.pt', 'alpha': 0.5,
                                    'vocabulary': str(root/'vocabulary.json')})
        write(root/'configs/stage_a.json', {'run': {'formal': {'candidate_epochs': [1]}}})
        write(root/'configs/stage_b.json', {'paths': {'corpus_root': 'training-corpus',
                                                   'stage_a_checkpoint': 'released_projector.pt'},
                                          'run': {'formal': {'candidate_epochs': [1, 2]}}})
        a = root/'stage_a/checkpoints/epoch_000001.ckpt'
        a.parent.mkdir(parents=True)
        torch.save({'state_dict': {'projector': torch.tensor([7.0])}}, a)
        write(root/'validation_stage_a/summary.json', {
            'status': 'complete', 'split': 'validation', 'evaluation_protocol': {'test_used': False},
            'candidates': {'A0_epoch_000001': {'checkpoint': str(a), 'metrics': {
                'listwise_kl': 0.1, 'true_pair_accuracy': 1.0, 'identity_accuracy': 1.0}}}})
        reproduce.select_a(self.args)
        for epoch in (1, 2):
            path = root/f'stage_b/checkpoints/epoch_{epoch:06d}.ckpt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'artifact_kind': 'candidate_checkpoint', 'candidate_epoch': epoch,
                        'state_dict': {'weight': torch.tensor([[float(epoch)]])}}, path)
        write(root/'configs/inference.json', {'paths': {'corpus_root': 'test-corpus'}})
        self.profile = {'profile_id': 'synthetic_test_profile', 'trajectory_batch': 10,
                        'adapter_options': {'pista': {'max_groups_per_generation_batch': 1}}}
        write(root/'configs/execution_profile.json', self.profile)
        formal = {'experiment_id': 'synthetic_handoff', 'model_order': ['top30'],
                  'models': {'top30': {'checkpoint': 'released.pt', 'alpha': 0.5}},
                  'conditions': ['reference', 'true', 'shuffled'],
                  'execution': {'profile_id': self.profile['profile_id']},
                  'protocol': {'axes': ['rdkit.ring_count'], 'draw_count': 100,
                               'sampling_steps': 50, 'top_k': [1], 'budgets': [1, 100],
                               'bootstrap_replicates': 10, 'bootstrap_seed': 1042,
                               'bootstrap_batch_size': 4,
                               'expected': {'selected_groups': 4, 'selected_cases': 2,
                                            'total_arms': 3,
                                            'trajectories_per_arm': 400, 'total_trajectories': 1200}}}
        write(root/'configs/formal_eval.json', formal)
        corpus = root/'data_package/data/evaluation/rdkit5_native/evaluation'
        corpus.mkdir(parents=True)
        rows = [self.frozen(index) for index in range(4)]
        (corpus/'validation_frozen.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        write(corpus/'validation_selection.json', {'split': 'validation', 'selected_group_count': 4,
                                                 'selected_case_count': 2})

    @staticmethod
    def frozen(index):
        seeds = root_draw_seeds_for_group(evaluation_group_index=index, draw_count=1)
        return {'schema_version': PISTA_FROZEN_SCHEMA, 'source_split': 'validation',
                'group_uid': f'group_{index}', 'case_id': f'case_{index//2}',
                'evaluation_group_index': index, 'legacy_group_idx': index,
                'canonical_product_smiles': 'CCO', 'branch': 'metric',
                'axis_id': 'rdkit.ring_count', 'polarity': -1 if index % 2 == 0 else 1,
                'true_guidance_text_id': f'true_{index}', 'true_guidance_text': 'Prefer this route.',
                'true_guidance_relation': 'canonical_true', 'shuffled_guidance_text_id': f'shuffled_{index}',
                'shuffled_mapping': {'recipient_group_uid': f'group_{index}', 'relation': 'different_axis',
                                     'donor_axis_id': 'rdkit.halogen_atom_count', 'donor_text_id': f'shuffled_{index}'},
                'positive_target_smiles': ['CCO'], 'negative_target_smiles': ['CCN'],
                'configured_rc_top_k': 5, 'effective_rc_k': 5,
                'rc_candidates': [{'rc_slot_index': slot, 'predictor_rank': slot+1,
                                   'predictor_score': 1.0/(slot+1), 'rc_permutation_key': f'slot_{slot}',
                                   'permutation': [0, 1, 2], 'is_padding_duplicate': slot > 0} for slot in range(5)],
                'root_draw_seeds': seeds, 'root_choice_indices': root_choice_indices_for_group(root_draw_seeds=seeds),
                'trajectory_seeds': trajectory_seeds_for_group(evaluation_group_index=index, draw_count=1)}

    def sampler_fixture(self, **kwargs):
        self.calls.append(kwargs)
        split = kwargs.get('split', 'test')
        config = json.loads(Path(kwargs['training_config']).read_text())
        profile = json.loads(Path(kwargs['execution_profile']).read_text())
        if split == 'validation':
            self.assertEqual(config['paths']['stage_a_checkpoint'], str(self.run/'stage_a/selected_projector.pt'))
        epoch = int(Path(kwargs['checkpoint']).stem.split('_')[-1]) if split == 'validation' else 1
        alpha = kwargs['guidance_alpha']
        positive_groups = (4 if epoch == 1 else 3) if alpha == 0.25 else 1
        rows = list(reproduce.read_rows(kwargs['frozen_manifest']))
        for arm, output in kwargs['arm_outputs'].items():
            generated = []
            for index, frozen in enumerate(rows):
                positive = arm == 'true' and index < positive_groups
                count = kwargs['expected_draw_count']
                generated.append({**frozen, 'arm': arm,
                    'generated_smiles': ['CCO' if positive else 'CCN']*count,
                    'categories': ['positive' if positive else 'negative']*count,
                    'rc_permutation_keys': [f'slot_{slot}' for slot in frozen['root_choice_indices']],
                    'guidance_residual_scale': None if arm == 'reference' else alpha,
                    'execution_profile_id': profile['profile_id'], 'sampling_steps': kwargs['sampling_steps'],
                    'per_sample_transition_kernel': 'stateless_counter_v1',
                    'sampling_rng_mode': 'paired_stateless_counter_v1'})
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(''.join(json.dumps(row)+'\n' for row in reversed(generated)))
        return {}

    def validate(self):
        flow.prepare_b_validation(self.args)
        with patch('retro_guided_generation.evaluation.pista.generate_pista_arms', side_effect=self.sampler_fixture):
            flow.validate_b(self.args)

    def test_stage_a_to_stage_b_selection_deployment_and_scoring(self):
        import torch
        self.validate()
        flow.select_b(self.args)
        prepared = json.loads((self.run/'prepared.json').read_text())
        self.assertEqual((prepared['selected_stage_b_epoch'], prepared['alpha']), (1, 0.25))
        self.assertEqual(Path(prepared['checkpoint']), self.run/'deployment/stage_b.pt')
        payload = torch.load(prepared['checkpoint'], weights_only=True)
        self.assertEqual(payload['state_dict']['weight'].item(), 1.0)
        with patch('retro_guided_generation.training.pista._build_pista_stage',
                   return_value=(torch.nn.Linear(1, 1, bias=False), {})):
            model = _load_pista_model(None, prepared['checkpoint'])
        self.assertEqual(model.weight.item(), 1.0)
        self.assertFalse(model.weight.requires_grad)
        self.assertEqual(torch.load(self.run/'deployment/stage_a.pt', weights_only=True)['state_dict']['projector'].item(), 7.0)
        test_rows = list(reproduce.read_rows(self.run/'validation_stage_b/frozen.jsonl.gz'))
        for row in test_rows:
            row['source_split'] = 'test'
        frozen = self.run/'test_frozen.jsonl.gz'
        inventory = self.run/'test_inventory.jsonl.gz'
        reproduce.write_rows(frozen, test_rows)
        reproduce.write_rows(inventory, [{'case_id': f'case_{i}', 'unique_canonical_endpoints': ['CCO', 'CCN']} for i in range(2)])
        prepared.update(frozen_manifest=str(frozen), endpoint_inventory=str(inventory), groups=4)
        reproduce.write_json(self.run/'prepared.json', prepared)
        with patch('retro_guided_generation.evaluation.pista.generate_pista_arms', side_effect=self.sampler_fixture):
            reproduce.infer(self.args)
        self.assertEqual(self.calls[-1]['guidance_alpha'], 0.25)
        self.assertEqual(self.calls[-1]['checkpoint'], str(self.run/'deployment/stage_b.pt'))
        score_results.configure_paths(self.run)
        self.assertEqual(score_results.main(), 0)
        self.assertEqual(json.loads((self.run/'reports/final_audit.json').read_text())['status'], 'passed')

    def test_test_split_is_rejected_before_preparation(self):
        path = self.run/'data_package/data/evaluation/rdkit5_native/evaluation/validation_frozen.jsonl'
        path.write_text(path.read_text().replace('"validation"', '"test"'))
        with self.assertRaisesRegex(ValueError, 'test rows'):
            flow.prepare_b_validation(self.args)
        self.assertFalse((self.run/'validation_stage_b').exists())

    def test_missing_validation_output_cannot_deploy(self):
        self.validate()
        flow._coordinate(self.run/'validation_stage_b', 1, 0.25).unlink()
        with self.assertRaises(FileNotFoundError):
            flow.select_b(self.args)
        self.assertFalse((self.run/'deployment').exists())

    def test_modified_scalar_selection_cannot_deploy(self):
        self.validate()
        path = self.run/'validation_stage_b/summary.json'
        value = json.loads(path.read_text())
        value['selected']['epoch'] = 2
        reproduce.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'differs from saved'):
            flow.select_b(self.args)
        self.assertFalse((self.run/'deployment').exists())

    def test_started_test_run_cannot_change_deployment(self):
        self.validate()
        (self.run/'outputs').mkdir()
        with self.assertRaises(FileExistsError):
            flow.select_b(self.args)

    def test_selected_epoch_metadata_must_match(self):
        import torch
        self.validate()
        path = self.run/'stage_b/checkpoints/epoch_000001.ckpt'
        value = torch.load(path, weights_only=True)
        value['candidate_epoch'] = 2
        torch.save(value, path)
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            flow.select_b(self.args)
        self.assertFalse((self.run/'deployment').exists())

    def test_inference_and_scoring_reject_inconsistent_alpha(self):
        prepared = json.loads((self.run/'prepared.json').read_text())
        prepared['alpha'] = 0.25
        reproduce.write_json(self.run/'prepared.json', prepared)
        with self.assertRaisesRegex(ValueError, 'configurations disagree'):
            reproduce.infer(self.args)
        with self.assertRaisesRegex(ValueError, 'configurations disagree'):
            score_results.configure_paths(self.run)


if __name__ == '__main__':
    unittest.main(verbosity=2)
