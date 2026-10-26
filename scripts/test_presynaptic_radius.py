"""Scientific radius boundaries and training-data contract checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from data.presynaptic_radius import FORMAT, radius_windows, validate_radius_training


def test_radius_is_geodesic_inclusive_and_uncapped():
    # A 25-node cable folds back toward its seed. Euclidean selection would
    # include the tip, but its skeleton distance is 24 um.
    pos = np.zeros((26, 3))
    pos[:13, 0] = np.arange(13) * 1000
    pos[13:25, 0] = np.arange(11, -1, -1) * 1000
    pos[25] = [0, 1, 0]  # nearby, disconnected node
    edges = np.c_[np.arange(24), np.arange(1, 25)]
    offsets, members, valid = radius_windows(pos, edges, np.arange(26), [0], [0], 20000)
    np.testing.assert_array_equal(offsets, [0, 21])
    np.testing.assert_array_equal(members, np.arange(21))
    assert valid.tolist() == [True]


def test_unobserved_bridges_short_components_duplicate_edges_and_anchors():
    pos = np.c_[np.arange(6) * 1000, np.zeros((6, 2))]
    edges = np.array([[0, 1], [1, 0], [0, 1], [1, 2], [2, 3]])
    ptr, members, valid = radius_windows(pos, edges, [0, 2, 3, 5],
                                         [0, 0, 5, 0, -1, 1], [5000, 0, 0, 5001, 0, 0], 2000)
    np.testing.assert_array_equal(ptr, [0, 2, 4, 5, 5, 5, 5])
    np.testing.assert_array_equal(members, [0, 2, 0, 2, 5])
    assert valid.tolist() == [True, True, True, False, False, False]


def test_random_graph_matches_independent_shortest_paths():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(50, 3)) * 2000
    edges = np.c_[np.arange(1, 50), rng.integers(0, np.arange(1, 50))]
    observed = np.arange(0, 50, 2)
    lengths = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    graph = csr_matrix((np.r_[lengths, lengths],
                        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(50, 50))
    distances = dijkstra(graph, indices=observed)
    ptr, members, valid = radius_windows(pos, edges, observed, observed, np.zeros(len(observed)), 20000)
    assert valid.all()
    for i in range(len(observed)):
        assert set(members[ptr[i]:ptr[i+1]]) == set(observed[distances[i, observed] <= 20000])


def test_zero_length_edge_and_empty_input():
    ptr, members, valid = radius_windows(np.zeros((2, 3)), [[0, 1]], [0, 1], [0], [0], 1)
    assert set(members) == {0, 1}
    ptr, members, valid = radius_windows(np.empty((0, 3)), [], [], [], [], 20000)
    assert ptr.tolist() == [0] and not len(members) and not len(valid)


def test_source_anchor_eligibility_overrides_stale_global_cutoff():
    ptr, members, valid = radius_windows(np.zeros((1, 3)), [], [0], [0, 0], [4000, 1000],
                                         20000, match_cutoff_nm=2000, anchor_eligible=[True, False])
    assert ptr.tolist() == [0, 1, 1] and members.tolist() == [0]
    assert valid.tolist() == [True, False]


@pytest.mark.parametrize('radius', [0, -1, float('nan'), float('inf')])
def test_invalid_radius(radius):
    with pytest.raises(ValueError):
        radius_windows(np.zeros((1, 3)), [], [0], [0], [0], radius)


def test_training_contract():
    args = SimpleNamespace(dataset='presynaptic_cave', num_embeddings=None, window_nm=20000,
                           architecture='pointwise_mlp', spatial=False, position=False, lpe=False,
                           single_presynaptic_embedding=False, presynaptic_memmap_root=None,
                           embedding_augmentation=None)
    meta = dict(format=FORMAT, complete=True, missing_root_ids=[], window_nm=20000)
    validate_radius_training(args, meta)
    for changes in ({'window_nm': 10000}, {'num_embeddings': 10}, {'architecture': 'graph_transformer'},
                    {'single_presynaptic_embedding': True}):
        with pytest.raises(ValueError):
            validate_radius_training(SimpleNamespace(**(vars(args) | changes)), meta)
    for changes in ({'window_nm': 10000}, {'format': 'fixed-k'}, {'complete': False}):
        with pytest.raises(ValueError):
            validate_radius_training(args, meta | changes)


@pytest.mark.parametrize('flag', ['cave_within_2um', 'cave_within_match_cutoff'])
def test_builder_preserves_source_anchor_policy_and_raw_vectors(tmp_path, flag):
    from scripts.prepare_presynaptic_radius import build_cell
    source = tmp_path / 'source.npz'
    output = tmp_path / 'radius'
    (output / 'cells').mkdir(parents=True)
    (output / 'records').mkdir()
    vectors = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)
    arrays = dict(cave_pos_nm=np.array([[0, 0, 0], [10000, 0, 0], [20000, 0, 0]]),
                  cave_edges=np.array([[0, 1], [1, 2]]), cave_observed_node_ids=np.arange(3),
                  cave_observed_x=vectors, cave_synapse_id=np.array([123, 456]),
                  cave_synapse_xyz_nm=np.array([[0, 4000, 0], [0, 1000, 0]]),
                  cave_nearest_observed_node=np.array([0, 0]),
                  cave_nearest_observed_distance_nm=np.array([4000, 1000]),
                  cave_valid_k_window=np.array([True, False]),
                  feature_run_id=np.asarray('latest'), feature_checkpoint_sha256=np.asarray('weights'))
    arrays[flag] = np.array([True, False])
    np.savez(source, **arrays)
    result = build_cell(('1', source, output, 20000, 2000, 'latest', 'weights'))
    assert result['valid'] == 1
    with np.load(output / 'cells/1.npz') as z:
        np.testing.assert_array_equal(z['cave_observed_x'], vectors)
        np.testing.assert_array_equal(z['cave_synapse_id'], [123, 456])
        np.testing.assert_array_equal(z['cave_window_offsets'], [0, 3, 3])
        np.testing.assert_array_equal(z['cave_window_members'], [0, 1, 2])
