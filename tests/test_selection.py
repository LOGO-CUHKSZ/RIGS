"""Synthetic checks for the validation-only Stage-A handoff."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import select_a, write_json


class StageASelectionTest(unittest.TestCase):
    def fixture(self, root):
        import torch
        write_json(root/'prepared.json', {})
        write_json(root/'configs/stage_a.json', {'run': {'formal': {'candidate_epochs': [1, 2]}}})
        write_json(root/'configs/stage_b.json', {'paths': {'stage_a_checkpoint': 'released.pt'}})
        candidates = {}
        for epoch, kl in [(1, 0.3), (2, 0.1)]:
            path = root/f'stage_a/checkpoints/epoch_{epoch:06d}.ckpt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': {'weight': torch.tensor([float(epoch)])}}, path)
            candidates[f'A0_epoch_{epoch:06d}'] = {
                'checkpoint': str(path), 'metrics': {
                    'listwise_kl': kl, 'true_pair_accuracy': 0.5, 'identity_accuracy': 0.0}}
        summary = {'status': 'complete', 'split': 'validation',
                   'evaluation_protocol': {'test_used': False}, 'candidates': candidates}
        write_json(root/'validation_stage_a/summary.json', summary)
        return summary

    def test_selection_hands_off_validation_winner_and_refuses_started_stage_b(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            args = SimpleNamespace(run_dir=root)
            (root/'stage_b').mkdir()
            with self.assertRaises(FileExistsError):
                select_a(args)
            (root/'stage_b').rmdir()
            select_a(args)
            config = json.loads((root/'configs/stage_b.json').read_text())
            payload = torch.load(config['paths']['stage_a_checkpoint'], weights_only=True)
            self.assertEqual(payload['state_dict']['weight'].item(), 2.0)
            self.assertEqual(json.loads((root/'stage_a/selection.json').read_text())['selected'], 'A0_epoch_000002')

    def test_test_data_and_incomplete_validation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.fixture(root)
            summary['evaluation_protocol']['test_used'] = True
            write_json(root/'validation_stage_a/summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'test data'):
                select_a(SimpleNamespace(run_dir=root))
            summary['evaluation_protocol']['test_used'] = False
            summary['candidates'].pop('A0_epoch_000002')
            write_json(root/'validation_stage_a/summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'all registered'):
                select_a(SimpleNamespace(run_dir=root))


if __name__ == '__main__':
    unittest.main(verbosity=2)
