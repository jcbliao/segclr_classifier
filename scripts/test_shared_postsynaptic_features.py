"""Shared features preserve distinct eligibility, row order and legacy loading."""
import numpy as np
import pytest

from data.postsynaptic_cache import load_postsynaptic_features
from scripts.share_postsynaptic_features import share


def test_shared_features_exact_and_resumable(tmp_path):
    a = tmp_path/'cave_skeletons_pre_post/k10/conf0/postsynaptic/cells'
    b = tmp_path/'native_skeletons_pre_post/scale16/k17/conf0/cells'
    vectors = np.arange(4*64, dtype=np.float32).reshape(4,64)
    originals = [vectors[[2,0,1]], vectors[[0,2,3]]]
    for directory, values, ids, eligible in zip(
            [a,b], originals, [[12,10,11],[10,12,13]], [[True,False,True],[True,True,False]]):
        directory.mkdir(parents=True)
        np.savez_compressed(directory/'123.npz', synapse_ids=ids,
                            eligible=eligible, post_x=values)
        np.testing.assert_array_equal(load_postsynaptic_features(directory/'123.npz'), values)
    share(tmp_path)
    share(tmp_path)
    for directory, values, ids, eligible in zip(
            [a,b], originals, [[12,10,11],[10,12,13]], [[True,False,True],[True,True,False]]):
        path = directory/'123.npz'
        np.testing.assert_array_equal(load_postsynaptic_features(path), values)
        with np.load(path) as z:
            assert 'post_x' not in z
            np.testing.assert_array_equal(z['synapse_ids'], ids)
            np.testing.assert_array_equal(z['eligible'], eligible)
    assert len(list((tmp_path/'postsynaptic_features/cells').glob('*.npz'))) == 1
    with np.load(a/'123.npz') as z:
        arrays = {key:z[key] for key in z.files}
    arrays['feature_rows'] = np.array([-1,0,0])
    np.savez_compressed(a/'123.npz', **arrays)
    with pytest.raises(ValueError, match='alignment'):
        load_postsynaptic_features(a/'123.npz')
