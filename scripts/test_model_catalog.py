"""Protect experiment namespaces and author-supplied artifact metadata."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.build_model_catalog import build


class ModelCatalogTests(unittest.TestCase):
    def test_namespaces_final_reports_and_publication_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'analysis').mkdir()
            payload = dict(args={'architecture': 'mean', 'num_embeddings': 20},
                           classes=['a', 'b'], best_epoch=0,
                           window_test_metrics={'macro_f1': 0.5, 'balanced_accuracy': 0.6},
                           test_metrics={'macro_f1': 0.7, 'balanced_accuracy': 0.8})
            for experiment in ('all_windows', 'presynaptic/cave_skeletons'):
                result = root / 'results' / experiment
                result.mkdir(parents=True)
                (result / 'same_run.json').write_text(json.dumps(payload))
                (result / 'same_run').mkdir()
                (result / 'same_run/best_metrics.json').write_text(json.dumps(payload))
            self.assertEqual(build(root), 2)
            with (root / 'results/summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['model_id'] for r in rows},
                             {'all_windows/same_run', 'presynaptic/cave_skeletons/same_run'})
            artifact_path = root / 'models/all_windows/same_run/artifact.json'
            artifact = json.loads(artifact_path.read_text())
            artifact.update(checkpoint_url='https://example.org/verified.pt', training_commit='original-commit')
            artifact_path.write_text(json.dumps(artifact))
            report = root / 'results/all_windows/same_run.json'
            payload['window_test_metrics']['macro_f1'] = 0.9
            report.write_text(json.dumps(payload))
            self.assertEqual(build(root), 2)
            regenerated = json.loads(artifact_path.read_text())
            self.assertEqual(regenerated['checkpoint_url'], artifact['checkpoint_url'])
            self.assertEqual(regenerated['training_commit'], artifact['training_commit'])
            self.assertEqual(regenerated['source_report_sha256'], hashlib.sha256(report.read_bytes()).hexdigest())
            card = (root / 'models/all_windows/same_run/README.md').read_text()
            self.assertIn('Best epoch as recorded: `0`', card)
            self.assertIn('0.9000', card)


if __name__ == '__main__':
    unittest.main()
