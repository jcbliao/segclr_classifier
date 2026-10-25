"""Canonical naming preserves all legacy data paths and frozen bytes."""
import hashlib
import json
from pathlib import Path

from scripts.migrate_morpho_naming import migrate, python_source


def test_migration_preserves_bytes_and_legacy_paths(tmp_path):
    folder=tmp_path/'analysis/casey_experiment'
    folder.mkdir(parents=True)
    (folder/'casey_analysis.py').write_text('"""Casey coarse types."""\nCOLUMN="casey_coarse"\n')
    frozen=folder/'plan.json'
    frozen.write_text('{"dataset":"casey_cohort","sha256":"abc"}')
    digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    dry=migrate(tmp_path)
    assert len(dry['renames'])==2 and frozen.exists()
    migrate(tmp_path,apply=True)
    canonical=tmp_path/'analysis/morpho_experiment'
    assert canonical.is_dir() and folder.is_symlink()
    assert (canonical/'casey_analysis.py').is_symlink()
    assert (folder/'casey_analysis.py').resolve()==canonical/'morpho_analysis.py'
    assert 'Morpho coarse types' in (canonical/'morpho_analysis.py').read_text()
    assert '"casey_coarse"' in (canonical/'morpho_analysis.py').read_text()
    assert hashlib.sha256((canonical/'plan.json').read_bytes()).hexdigest()==digest
    assert not migrate(tmp_path,apply=True)['renames']


def test_provenance_keys_and_attribution_preserved_but_paths_and_imports_renamed():
    code='''# Casey models
from scripts.build_casey_folds import run
path="analysis/casey_confidence/report.json"
external="/orcd/scratch/lab/casey_cohort"
key="casey_coarse_confidence"
author="Casey Schneider-Mizell"
title="Casey coarse confidence"
'''
    result=python_source(code,{'build_casey_folds':'build_morpho_folds'})
    assert 'from scripts.build_morpho_folds' in result
    assert 'analysis/morpho_confidence/report.json' in result
    assert '/orcd/scratch/lab/casey_cohort' in result
    assert 'key="casey_coarse_confidence"' in result
    assert 'Casey Schneider-Mizell' in result
    assert 'title="Morpho coarse confidence"' in result
