"""Focused checks for config precedence and validation, without training imports."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from scripts.training_config import parse_training_args


class TrainingConfigTests(unittest.TestCase):
    def parse(self, values, flags=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(values))
            parser = argparse.ArgumentParser()
            parser.add_argument('--architecture', choices=['mean', 'mpnn'], default='mpnn')
            parser.add_argument('--epochs', type=int, default=16)
            parser.add_argument('--amp', action='store_true')
            parser.add_argument('--cls-resnet', action=argparse.BooleanOptionalAction, default=True)
            parser.add_argument('--num-embeddings', type=int, default=20)
            parser.add_argument('--radius-dataset', dest='num_embeddings', action='store_const', const=None)
            return parse_training_args(parser, ['--config', str(path), *flags])

    def test_defaults_and_cli_precedence(self):
        args = self.parse({'architecture': 'mean', 'epochs': 4, 'amp': True}, ['--epochs', '8'])
        self.assertEqual((args.architecture, args.epochs, args.amp), ('mean', 8, True))
        json.dumps(vars(args))

    def test_invalid_settings_are_rejected(self):
        for values in ({'unknown': 1}, {'architecture': 'other'}, {'amp': 'false'},
                       {'epochs': True}, {'epochs': 2.5}, {'epochs': None},
                       {'cls_resnet': 'false'}, {'num_embeddings': 'ten'}, []):
            with self.subTest(values=values), self.assertRaises(SystemExit):
                self.parse(values)

    def test_boolean_override_and_shared_destination(self):
        args = self.parse({'cls_resnet': True, 'num_embeddings': 10}, ['--no-cls-resnet'])
        self.assertEqual((args.cls_resnet, args.num_embeddings), (False, 10))
        self.assertIsNone(self.parse({'num_embeddings': None}).num_embeddings)

    def test_no_config_preserves_cli(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=16)
        self.assertEqual(parse_training_args(parser, ['--epochs', '3']).epochs, 3)


if __name__ == '__main__':
    unittest.main()
