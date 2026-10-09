"""JSON defaults for the existing training CLI; explicit flags take precedence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_training_args(parser: argparse.ArgumentParser, argv=None):
    parser.add_argument('--config', type=Path, help='JSON object of training argument defaults; CLI flags override it')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    known, _ = probe.parse_known_args(argv)
    if known.config:
        try:
            values = json.loads(known.config.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f'cannot read config: {exc}')
        if not isinstance(values, dict):
            parser.error('config must be a JSON object')
        actions = {}
        for action in parser._actions:
            if action.dest not in ('help', 'config'):
                # A hidden store_const alias may share a destination with a
                # typed option (e.g. --radius-dataset / --num-embeddings).
                if action.dest not in actions or isinstance(action, argparse._StoreAction):
                    actions[action.dest] = action
        unknown = values.keys() - actions.keys()
        if unknown:
            parser.error(f'unknown config keys: {", ".join(sorted(unknown))}')
        for key, value in values.items():
            action = actions[key]
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                   argparse.BooleanOptionalAction)):
                if not isinstance(value, bool):
                    parser.error(f'{key} must be a JSON boolean')
            elif value is not None and action.type:
                try:
                    if action.type in (int, float) and isinstance(value, bool):
                        raise ValueError('boolean is not a number')
                    if action.type is int and not isinstance(value, int):
                        raise ValueError('expected integer')
                    values[key] = action.type(value)
                except (TypeError, ValueError) as exc:
                    parser.error(f'invalid config value for {key}: {exc}')
            elif value is not None and isinstance(action.default, str) and not isinstance(value, str):
                parser.error(f'{key} must be a JSON string')
            if action.choices is not None and values[key] not in action.choices:
                parser.error(f'invalid config value for {key}: {values[key]!r}')
            if value is None and action.default is not None and key != 'num_embeddings':
                parser.error(f'{key} cannot be null')
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    # Reports store resolved arguments, so they can themselves be replayed as
    # configs without an extra config-file metadata key.
    del args.config
    return args
