"""Write every local head's scores on held-out presynaptic windows."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from scripts.prepare_presynaptic_radius import save, sha
from scripts.predict_incoming_fragment_cell_types import load_checkpoint


def score_columns(distribution, head_nodes):
    """Keep conditional local scores distinct from joint level probabilities."""
    columns = {}
    for i, (spec, logits) in enumerate(zip(head_nodes, distribution['head_logits'], strict=True)):
        if logits.ndim != 2 or logits.shape[1] != spec['n_children'] or not torch.isfinite(logits).all():
            raise ValueError('Invalid local logits')
        for suffix, values in [('logits', logits), ('probabilities', logits.softmax(-1))]:
            columns[f'head_{i}_{suffix}'] = pa.FixedSizeListArray.from_arrays(
                pa.array(values.detach().float().cpu().numpy().ravel()), spec['n_children'])
        columns[f'head_{i}_prediction'] = pa.array(logits.argmax(-1).detach().cpu().numpy())
    for level, values in enumerate(distribution['level_probabilities']):
        columns[f'level_{level}_joint_probabilities'] = pa.FixedSizeListArray.from_arrays(
            pa.array(values.detach().float().cpu().numpy().ravel()), values.shape[1])
    predictions = distribution['predictions'].detach().cpu().numpy()
    columns['top_down_predictions'] = pa.FixedSizeListArray.from_arrays(pa.array(predictions.ravel()), predictions.shape[1])
    return columns


def export_fold(state, record, dataset):
    checkpoints = list(Path(record['results_dir']).glob('*/checkpoint_best.pt'))
    if len(checkpoints) != 1:
        raise ValueError(f'Expected one completed radius checkpoint: {checkpoints}')
    path = checkpoints[0]
    model, manifest = load_checkpoint(path)
    model = model.cuda().eval()
    if model.cls_head.hierarchy.level_classes != dataset.hierarchy.level_classes:
        raise ValueError('Classifier hierarchy differs from the export dataset')
    heads = model.cls_head.nodes
    folder = state / 'all_lcpn_predictions'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"fold{record['fold']}.parquet"
    marker = target.with_suffix('.json')
    identity = dict(checkpoint=str(path), checkpoint_sha256=sha(path), manifest_sha256=sha(manifest),
                    plan_sha256=sha(state / 'plan.json'), fold=record['fold'], windows=len(dataset),
                    level_classes=dataset.hierarchy.level_classes, local_heads=heads,
                    score_semantics='all local heads on every row; probabilities conditional on parent; joint level probabilities separate',
                    split='held_out_fold; same fold used for checkpoint selection',
                    deduplication='one representative synapse per per-cell membership set, matching training')
    if marker.exists():
        previous = json.loads(marker.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing classification output has different provenance')
        if target.exists() and previous['parquet_sha256'] == sha(target):
            print('Completed all-head classification already exists', target, flush=True)
            return
        raise ValueError('Completed classification output changed or is missing')
    temporary = target.with_suffix('.tmp.parquet')
    writer = None
    confusion = [np.zeros((head['n_children'], head['n_children']), np.int64) for head in heads]
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(dataset), 512), desc='All LCPN heads', mininterval=10):
                indices = np.arange(start, min(start + 512, len(dataset)))
                batch = dataset.__getitems__(indices).to('cuda')
                scores = model.predict_distribution(batch.x, batch.edge_index, batch.batch,
                                                     postsynaptic_embedding=batch.postsynaptic_embedding)
                targets = batch.y_levels.cpu().numpy()
                for i, head in enumerate(heads):
                    routed = (np.ones(len(indices), bool) if head['parent_level'] < 0 else
                              targets[:, head['parent_level']] == head['parent_global_idx'])
                    child_ids = head['child_global_indices']
                    mapping = {value: j for j, value in enumerate(child_ids)}
                    truth = np.array([mapping[int(v)] for v in targets[routed, head['child_level']]], np.int64)
                    predictions = scores['head_logits'][i].argmax(-1).cpu().numpy()[routed]
                    np.add.at(confusion[i], (truth, predictions), 1)
                columns = dict(root_id=pa.array(dataset.index_root_ids[indices]),
                               synapse_id=pa.array(batch.synapse_id.cpu().numpy()),
                               window_row=pa.array(dataset.index_window_rows[indices]),
                               n_embeddings=pa.array(dataset.window_sizes[indices]))
                columns.update(score_columns(scores, heads))
                table = pa.table(columns)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema.with_metadata(
                        {b'provenance': json.dumps(identity).encode()}), compression='zstd')
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None or pq.ParquetFile(temporary).metadata.num_rows != len(dataset):
        raise ValueError('Incomplete prediction output')
    temporary.replace(target)
    metrics = [dict(parent=head['parent_name'], children=head['children'],
                    support=int(matrix.sum()), accuracy=float(np.trace(matrix) / matrix.sum()) if matrix.sum() else None,
                    confusion_matrix=matrix.tolist()) for head, matrix in zip(heads, confusion, strict=True)]
    save(marker, dict(identity, parquet_sha256=sha(target),
                      local_head_metrics_conditioned_on_true_parent=metrics))
    print('ALL LCPN CLASSIFICATION SAVED', target, len(dataset), len(heads), flush=True)
