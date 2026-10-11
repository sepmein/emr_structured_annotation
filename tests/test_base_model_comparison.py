"""Offline guardrails for comparable saved encoder experiments."""
import tempfile
from pathlib import Path
import unittest

from emr_annotation.evaluation.base_model_comparison import FROZEN_FILES, COMMON_PARAMETERS, validate_contract


class BaseModelComparisonTests(unittest.TestCase):
    def fixture(self, root):
        dirs = [root / 'baseline', root / 'candidate']
        for directory in dirs:
            directory.mkdir()
            for name in FROZEN_FILES:
                (directory / name).write_text('synthetic frozen input ' + name, encoding='utf-8')
        receipt = {'parameters': {name: 1 for name in COMMON_PARAMETERS},
                   'checkpoint_reloaded': True, 'all_checkpoint_tensors_finite': True,
                   'status': 'completed_synthetic'}
        return dirs, receipt

    def test_different_reference_or_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dirs, r = self.fixture(Path(temp))
            checks = validate_contract(*dirs, r, r)
            self.assertEqual(set(checks), set(FROZEN_FILES))
            (dirs[1] / 'test.json').write_text('different synthetic answer', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'test.json'):
                validate_contract(*dirs, r, r)

    def test_different_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dirs, r = self.fixture(Path(temp))
            other = {**r, 'parameters': {**r['parameters'], 'threshold': .9}}
            with self.assertRaisesRegex(ValueError, 'threshold'):
                validate_contract(*dirs, r, other)

    def test_unloaded_or_failed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dirs, r = self.fixture(Path(temp))
            for bad in ({**r, 'checkpoint_reloaded': False}, {**r, 'status': 'failed'}):
                with self.assertRaises(ValueError):
                    validate_contract(*dirs, r, bad)


if __name__ == '__main__':
    unittest.main()
