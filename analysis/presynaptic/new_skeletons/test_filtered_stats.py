import numpy as np
import filtered_stats as filtered
import skeleton_stats as shared


def test_component_threshold_and_upstream_population(tmp_path):
    original_pos = np.array([[0,0,0],[6000,0,0],[7000,0,0],
                             [0,6000,0],[0,7000,0]], dtype=np.float32)
    original_edges = np.array([[0,1],[1,2],[0,3],[3,4]])
    original = tmp_path / 'original.npz'
    np.savez(original, vertices=original_pos, edges=original_edges)
    cell_path = tmp_path / 'cell.npz'
    # First component has exactly 5 unique sites plus a duplicate; second has 6.
    ids = np.r_[np.arange(11), 0, 99]
    nodes = np.r_[np.zeros(5, int), np.full(6, 2), 0, 0]
    accepted = np.ones(len(ids), bool); accepted[-1] = False
    np.savez(cell_path, new_pos_nm=original_pos[1:], new_edges=np.array([[0,1],[2,3]]),
             new_edge_length_nm=np.array([1000,1000]), root_xyz_nm=np.zeros(3),
             new_synapse_id=ids, new_nearest_observed_node=nodes, new_within_2um=accepted)
    with np.load(cell_path) as cell:
        sites, _ = shared.mapped_presynaptic_counts(cell, 4)
        keep = shared.component_node_mask(4, cell['new_edges'], sites, 5)
        assert keep.tolist() == [False, False, True, True]
    lengths, branches, synapses, stats = shared.derived_edge_upstream_counts(
        cell_path, original, 5000, min_component_sites=5)
    assert lengths.tolist() == [1000]
    assert branches.tolist() == [0]
    assert synapses.tolist() == [6]
    assert stats['components'] == 1 and stats['included_edges'] == 1
    assert stats['matched_unique_presynaptic_sites'] == 6


def test_summary_reuses_cache(tmp_path, monkeypatch):
    path = tmp_path / 'edge_lengths_nm.float32'
    np.arange(1, 101, dtype=np.float32).tofile(path)
    (tmp_path / 'complete.json').write_text('{}')
    values = np.memmap(path, dtype=np.float32, mode='r')
    first = filtered.summarize(values)
    def unexpected_scan(*args, **kwargs):
        raise AssertionError('Cached summary must not rescan edge data')
    monkeypatch.setattr(shared, 'summarize', unexpected_scan)
    second = filtered.summarize(values)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


def test_plot_cache_avoids_scans_and_refits(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    path = tmp_path / 'edge_lengths_nm.float32'
    np.linspace(80, 500, 2000, dtype=np.float32).tofile(path)
    (tmp_path / 'complete.json').write_text('{}')
    values = np.memmap(path, dtype=np.float32, mode='r')
    for log_space in (True, False):
        fig, first = filtered.plot_distribution_and_gmm(
            values, (80., 500.), log_space, n_components=1)
        plt.close(fig)
        with monkeypatch.context() as patch:
            def unexpected(*args, **kwargs):
                raise AssertionError('Warm plotting must not scan or refit')
            patch.setattr(shared, 'streaming_histogram', unexpected)
            patch.setattr(shared, 'fit_histogram_gmm', unexpected)
            fig, second = filtered.plot_distribution_and_gmm(
                values, (80., 500.), log_space, n_components=1)
            np.testing.assert_allclose(first['means'], second['means'])
            plt.close(fig)


def test_recorded_cohort_excludes_later_cave_only_additions(tmp_path):
    import json
    cells = tmp_path/'cells'
    cells.mkdir()
    for rid in (1,2,3):
        (cells/f'{rid}.npz').touch()
    (tmp_path/'metadata.json').write_text(json.dumps({'discarded_without_current_embeddings_root_ids':[2]}))
    assert [p.stem for p in shared.presynaptic_cell_paths(tmp_path)] == ['1','3']
