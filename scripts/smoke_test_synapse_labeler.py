"""Offline checks for synapse polarity, snapshot resolution, and viewer layer isolation."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd

from make_synapse_labeler import collect_fragments, make_state


class SynapseLabelerTests(unittest.TestCase):
    def test_partner_selection_deduplication_and_unresolved(self):
        fragment = (2 << 56) + 101
        client = SimpleNamespace(
            materialize=SimpleNamespace(get_timestamp=Mock(return_value="snapshot")),
            chunkedgraph=SimpleNamespace(get_roots=Mock(return_value=[fragment])),
        )
        rows = pd.DataFrame([
            dict(synapse_id=i, cell_root_id=999, partner_root_id=partner,
                 partner_supervoxel_id=123, ctr_x_nm=100, ctr_y_nm=200, ctr_z_nm=300)
            for i, partner in enumerate([888, 888, 0])
        ])
        ids, records = collect_fragments(client, rows)
        self.assertEqual(ids, [fragment])
        self.assertEqual(records[0]['l2_ids'], [str(fragment)])
        self.assertEqual(records[-1]['l2_ids'], [])
        client.chunkedgraph.get_roots.assert_called_once_with(
            [123], stop_layer=2, timestamp="snapshot")

    def test_mesh_only_layer_and_uint64_precision(self):
        root = 864691135943534196
        state = make_state(root, [101, 102], [800, 1600, 4000], '#123456')
        normal, fragments, incoming = state['layers'][1:]
        self.assertEqual(normal['segments'], [str(root)])
        self.assertTrue(normal['source'][0]['subsources']['graph'])
        self.assertEqual(fragments['segments'], ['101', '102'])
        self.assertEqual(fragments['segmentColors'], {'101': '#123456', '102': '#123456'})
        self.assertFalse(fragments['source'][0]['subsources']['graph'])
        self.assertFalse(fragments['source'][0]['subsources']['default'])
        self.assertTrue(fragments['source'][0]['subsources']['mesh'])
        self.assertEqual(state['position'], [100, 200, 100])

    def test_incoming_partner_and_separate_layer(self):
        fragment = (2 << 56) + 102
        client = SimpleNamespace(
            materialize=SimpleNamespace(get_timestamp=lambda: "snapshot"),
            chunkedgraph=SimpleNamespace(get_roots=Mock(return_value=[fragment])),
        )
        rows = pd.DataFrame([dict(synapse_id=1, cell_root_id=999,
                                 partner_root_id=777, partner_supervoxel_id=456,
                                 ctr_x_nm=100, ctr_y_nm=200, ctr_z_nm=300)])
        ids, records = collect_fragments(client, rows, "incoming")
        self.assertEqual(records[0]['presynaptic_root_id'], '777')
        state = make_state(999, [101], incoming_l2_ids=ids)
        outgoing, incoming = state['layers'][2:]
        self.assertEqual(outgoing['segments'], ['101'])
        self.assertEqual(incoming['segments'], [str(fragment)])
        self.assertNotEqual(outgoing['segmentDefaultColor'], incoming['segmentDefaultColor'])
        self.assertTrue(incoming['visible'])
        self.assertFalse(incoming['source'][0]['subsources']['graph'])

    def test_empty(self):
        self.assertEqual(collect_fragments(None, pd.DataFrame()), ([], []))
        self.assertEqual(make_state(999, [])['layers'][2]['segments'], [])


if __name__ == '__main__':
    unittest.main()
