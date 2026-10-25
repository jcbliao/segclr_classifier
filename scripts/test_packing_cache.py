import numpy as np
import pytest
import torch
from data.packed_windows import PackedWholeCellWindows
from data.packing_cache import prepare_cached,cache_dir
from scripts.test_default_training_pipeline import whole_windows

def test_persistent_cpu_cache_roundtrip_and_repeated_windows(tmp_path,monkeypatch):
    monkeypatch.setenv('SEGCLR_PACKING_CACHE',str(tmp_path))
    original=PackedWholeCellWindows(whole_windows())
    prepare_cached(original,'test-key','train',workers=2,build=True)
    assert (cache_dir('test-key','train')/'complete.json').exists()
    restored=PackedWholeCellWindows(whole_windows())
    monkeypatch.setattr(restored,'prepare_packed',lambda **kw:pytest.fail('Repacked an existing cache'))
    assert prepare_cached(restored,'test-key','train')
    assert isinstance(restored._packed['fields']['x'],np.memmap)
    for key in original.get_batch([4,0,0,2]).keys():
        torch.testing.assert_close(original.get_batch([4,0,0,2])[key],restored.get_batch([4,0,0,2])[key],rtol=0,atol=0)

def test_gpu_cache_only_refuses_missing_and_incomplete_cache(tmp_path,monkeypatch):
    monkeypatch.setenv('SEGCLR_PACKING_CACHE',str(tmp_path))
    monkeypatch.setenv('SEGCLR_PACKING_CACHE_ONLY','1')
    ds=PackedWholeCellWindows(whole_windows())
    with pytest.raises(RuntimeError,match='not ready'):prepare_cached(ds,'missing','train')

def test_cache_identity_shares_heads_but_invalidates_features_and_input_changes(tmp_path):
    from scripts.train_gnn import build_training_parser
    from scripts.training_config import parse_training_args
    from data.packing_cache import identity
    args=parse_training_args(build_training_parser(),[])
    args.dataset='presynaptic_dense';args.architecture='mean';args.presynaptic_database=str(tmp_path)
    cells=tmp_path/'cells';cells.mkdir();source=cells/'11.npz';source.write_bytes(b'first')
    manifest={'cells':{'11':{'split':'train','cell_type':'pyramidal'}}}
    initial=identity(args,manifest)
    args.cls_resnet=not args.cls_resnet;args.architecture='pointwise_mlp'
    assert identity(args,manifest)==initial
    args.architecture='graph_transformer'
    assert identity(args,manifest)!=initial
    args.architecture='mean';source.write_bytes(b'changed source')
    assert identity(args,manifest)!=initial

@pytest.mark.parametrize('geometry',[False,True])
@pytest.mark.parametrize('single,post',[(False,False),(False,True),(True,True)])
def test_presynaptic_cache_preserves_geometry_posts_and_single_points(tmp_path,monkeypatch,geometry,single,post):
    from collections import OrderedDict
    from scripts.test_batched_presynaptic import fixture,same
    from data.packed_presynaptic import PackedPresynapticWindowDataset
    monkeypatch.setenv('SEGCLR_PACKING_CACHE',str(tmp_path))
    ds,h=fixture(single=single,post=post);ds.hierarchy=h;ds.geometry_required=geometry;ds._cache=OrderedDict()
    expected=ds.__getitems__([0,2,1,0,3])
    ds.__class__=PackedPresynapticWindowDataset
    prepare_cached(ds,'presyn','train',workers=2,build=True)
    same(expected,ds.__getitems__([0,2,1,0,3]))


def test_successful_training_removes_packed_cache(tmp_path, monkeypatch):
    from data.packing_cache import training_cache_session
    monkeypatch.setenv('SEGCLR_PACKING_CACHE', str(tmp_path))
    with training_cache_session():
        ds = PackedWholeCellWindows(whole_windows())
        prepare_cached(ds, 'completed', 'train', workers=1, build=True)
        assert cache_dir('completed', 'train').exists()
    assert not cache_dir('completed', 'train').parent.exists()
    # A later queued GPU consumer can rebuild an intentionally deleted cache.
    monkeypatch.setenv('SEGCLR_PACKING_CACHE_ONLY', '1')
    with training_cache_session():
        prepare_cached(PackedWholeCellWindows(whole_windows()), 'completed', 'train', workers=1)
    assert not cache_dir('completed', 'train').parent.exists()


def test_failed_training_keeps_cache_for_resume(tmp_path, monkeypatch):
    from data.packing_cache import training_cache_session
    monkeypatch.setenv('SEGCLR_PACKING_CACHE', str(tmp_path))
    with pytest.raises(RuntimeError, match='training interrupted'):
        with training_cache_session():
            prepare_cached(PackedWholeCellWindows(whole_windows()), 'resume', 'train', workers=1, build=True)
            raise RuntimeError('training interrupted')
    assert cache_dir('resume', 'train').exists()


def test_shared_cache_survives_until_last_trainer_finishes(tmp_path, monkeypatch):
    from data.packing_cache import training_cache_session
    monkeypatch.setenv('SEGCLR_PACKING_CACHE', str(tmp_path))
    with training_cache_session():
        prepare_cached(PackedWholeCellWindows(whole_windows()), 'shared', 'train', workers=1, build=True)
        with training_cache_session():
            prepare_cached(PackedWholeCellWindows(whole_windows()), 'shared', 'train')
        assert cache_dir('shared', 'train').exists()
    assert not cache_dir('shared', 'train').parent.exists()
