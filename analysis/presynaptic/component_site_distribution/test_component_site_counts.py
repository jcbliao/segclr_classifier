from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from analysis.presynaptic.component_site_distribution import component_site_counts as stats


def test_zero_components_duplicates_rejections_and_strict_cutoff(tmp_path):
    path=tmp_path/'123.npz'
    np.savez(path,new_pos_nm=np.zeros((5,3)),new_edges=np.array([[0,1],[2,3]]),
        new_synapse_id=np.array([1,1,2,3,4,5,6,7,8]),
        new_nearest_observed_node=np.array([0,0,0,0,0,0,2,2,4]),
        new_within_2um=np.array([True]*8+[False]))
    frame=stats.cell_components(('Non-native',path))
    assert frame.n_presynaptic_sites.tolist()==[5,2,0]
    assert frame.n_edges.tolist()==[1,1,0]
    assert frame.n_nodes.tolist()==[2,2,1]
    result=stats.cutoff_table(frame,cutoffs=[0,2,5])
    assert result.retained_components.tolist()==[2,1,0]
    assert result.retained_edges.tolist()==[2,1,0]
    summary=stats.summary_table(frame).iloc[0]
    assert summary.zero_site_components==1 and summary.components==3
