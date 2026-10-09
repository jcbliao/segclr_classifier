"""Analytic and brute-force checks of skeleton comparison geometry."""
import unittest
import numpy as np

from compare_skeleton_geometry import SegmentIndex, compare, geometry, topology


class GeometryTests(unittest.TestCase):
    def test_identity_and_vertex_density(self):
        sparse = geometry([[0, 0, 0], [10000, 0, 0]], [[0, 1]])
        vertices = np.column_stack((np.linspace(0, 10000, 101), np.zeros((101, 2))))
        dense = geometry(vertices, np.column_stack((np.arange(100), np.arange(1, 101))))
        result = compare(sparse, dense, 250, [250, 500])
        self.assertAlmostEqual(result['symmetric_mean_um'], 0, places=10)
        self.assertEqual(result['f1_0.25um'], 1)
        self.assertAlmostEqual(result['teasar_cave_cable_ratio'], 1)

    def test_parallel_offset(self):
        a = geometry([[0, 0, 0], [10000, 0, 0]], [[0, 1]])
        b = geometry([[0, 600, 0], [10000, 600, 0]], [[0, 1]])
        result = compare(a, b, 100, [500, 1000])
        self.assertAlmostEqual(result['symmetric_mean_um'], .6)
        self.assertEqual(result['f1_0.5um'], 0)
        self.assertEqual(result['f1_1um'], 1)

    def test_exact_segments_against_brute_force(self):
        rng = np.random.default_rng(18)
        vertices = rng.normal(size=(40, 3)) * 3000
        edges = np.arange(40).reshape(-1, 2)
        v, e, lengths = geometry(vertices, edges)
        points = rng.normal(size=(81, 3)) * 10000
        # Independent direct projection to every original segment.
        delta = v[e[:, 1]] - v[e[:, 0]]
        relative = points[:, None, :] - v[e[:, 0]]
        fraction = np.clip((relative * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0, 1)
        expected = np.linalg.norm(relative - fraction[:, :, None] * delta, axis=2).min(axis=1)
        actual = SegmentIndex(v, e, lengths).distances(points)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)

    def test_missing_cable_direction_and_bounds(self):
        full = geometry([[0, 0, 0], [10000, 0, 0]], [[0, 1]])
        half = geometry([[0, 0, 0], [5000, 0, 0]], [[0, 1]])
        result = compare(full, half, 100, [500])
        self.assertAlmostEqual(result['precision_0.5um'], 1)
        self.assertAlmostEqual(result['recall_0.5um'], .55)
        self.assertAlmostEqual(result['symmetric_mean_um'], .625)
        self.assertLessEqual(result['recall_lower_0.5um'], .55)
        self.assertGreaterEqual(result['recall_upper_0.5um'], .55)

    def test_duplicate_edges_and_topology(self):
        v, e, lengths = geometry([[0, 0, 0], [100, 0, 0], [0, 100, 0], [-100, 0, 0], [999, 999, 0]],
                                 [[0, 1], [1, 0], [0, 2], [0, 3], [0, 0]])
        result = topology(v, e, lengths)
        self.assertEqual(result['components'], 2)
        self.assertEqual(result['branch_nodes'], 1)
        self.assertEqual(result['endpoints'], 3)
        self.assertEqual(result['cycle_rank'], 0)
        self.assertAlmostEqual(result['cable_um'], .3)

    def test_empty_cable_is_explicit_error(self):
        skeleton = geometry([[0, 0, 0]], [])
        with self.assertRaises(ValueError):
            compare(skeleton, skeleton, 250, [250])

    def test_zero_length_connector_keeps_topology(self):
        skeleton = geometry([[0, 0, 0], [0, 0, 0], [1000, 0, 0]], [[0, 1], [1, 2]])
        self.assertEqual(topology(*skeleton)['components'], 1)
        self.assertEqual(topology(*skeleton)['isolated_nodes'], 0)
        result = compare(skeleton, skeleton, 100, [250])
        self.assertAlmostEqual(result['symmetric_mean_um'], 0)


if __name__ == '__main__':
    unittest.main()
