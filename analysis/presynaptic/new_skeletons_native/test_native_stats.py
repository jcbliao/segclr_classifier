from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data import build_native_presynaptic_stats as builder
from analysis.presynaptic.new_skeletons_native import skeleton_stats as native
from analysis.presynaptic.new_skeletons import skeleton_stats as shared


def test_native_mapping_filter_and_upstream_population(tmp_path, monkeypatch):
    source_db = tmp_path/'source'
    (source_db/'cells').mkdir(parents=True)
    monkeypatch.setattr(builder, 'SOURCE_DB', source_db)
    out = tmp_path/'native'
    for d in ('cells','audit'):
        (out/d).mkdir(parents=True)
    original = tmp_path/'1.npz'
    pos = np.array([[0,0,0],[6000,0,0],[6100,0,0],
                    [0,6000,0],[0,6100,0]], np.float32)
    np.savez(original,root_id=[1],vertices=pos,
             edges=np.array([[0,1],[1,2],[0,3],[3,4]]),spacing_nm=111.)
    # Exactly five unique accepted sites in branch one; six in branch two.
    # Duplicate ID zero must not qualify branch one. The final row is rejected
    # by distance but still belongs in diagnostics for retained branch two.
    ids = np.r_[np.arange(11),0,99]
    xyz = np.vstack([np.tile(pos[1],(5,1)),np.tile(pos[3],(6,1)),pos[1],[0,9000,0]])
    np.savez(source_db/'cells/1.npz',root_xyz_nm=pos[0],
             new_synapse_id=ids,new_synapse_xyz_nm=xyz)
    audit = builder.build_cell(('1',str(original),str(out)))
    assert audit['retained_components'] == 1
    assert audit['retained_edges'] == 1
    assert audit['retained_unique_accepted_sites'] == 6
    assert audit['rows_nearest_retained_component'] == 7
    assert audit['retained_rows_within_2um'] == 6
    with np.load(out/'cells/1.npz') as cell:
        assert cell['component_presynaptic_sites'].tolist()==[5,6]
    lengths, branches, sites, stats = shared.derived_edge_upstream_counts(
        out/'cells/1.npz', original, 5000., min_component_sites=5)
    assert lengths.tolist()==[100.]
    assert branches.tolist()==[0]
    assert sites.tolist()==[6]
    monkeypatch.setattr(native,'CACHE_ROOT',tmp_path/'cache')
    retained=native.load_presynaptic_lengths(out)
    assert retained.tolist()==[100.]
