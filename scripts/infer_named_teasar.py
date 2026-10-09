"""SegCLR inference at named TEASAR nodes, isolated from CAVE embeddings."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import logging
import numpy as np
import torch
from segclr_db import store as st
from segclr_db.skeletons import SkeletonCache
from segclr.inference import SegCLRInference
from segclr.inference.inference import EM_PATH, EM_SCALE, SEGMENTATION, BOX_SIZE
from segclr.data.crops import CropLoader
from segclr.utils.checkpointing import load_checkpoint_automatic
from types import SimpleNamespace

BASE = Path(os.environ.get('INFERENCE_BASE', '/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910_resumed'))
NAME = os.environ.get('SKELETON_NAME', 'teasar_testing_111nm_20260910')
COMMIT_LOCK = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910_resumed/.named_embedding_write.lock')
EXPERIMENT = 'resnet_860b_reshuffled'
CHECKPOINT = 'checkpoint_e0_s95000'
RUN = 'resnet_860b_reshuffled__20260603_150412'
CHECKPOINT_PATH = Path('/orcd/data/sdorkenw/001/collina/segclr_runs/resnet_860b_reshuffled_20260603_150412/checkpoints/checkpoint_e0_s95000.pt')


class NamedInference(SegCLRInference):
    """Use the established crop/batch pipeline with an explicitly pinned model.

    The generic constructor resolves all registry runs and creates a CAVE-node
    writer. Neither is used here: this task supplies a fixed checkpoint and
    its own skeleton-name-scoped writer.
    """
    def __init__(self):
        self.batch_size=32
        self.num_threads=8
        self.precision='tf32'
        self.channels_last=True
        self.allow_partial=False
        self.device=torch.device('cuda')
        self.mat_version=1718
        self._apply_precision()
        seg_path,seg_scale=SEGMENTATION[1718]
        self.crops=CropLoader(em_path=EM_PATH,seg_path=seg_path,scale=EM_SCALE,
                             seg_scale=seg_scale,box_size=BOX_SIZE,oob='pad')
        self.model=load_checkpoint_automatic(CHECKPOINT_PATH,self.device)
        self.model.eval()
        self.model=self.model.to(memory_format=torch.channels_last_3d)
        assert self.model.bottleneck_dim==64
        self.experiment=SimpleNamespace(embedding_dim=64)
        self._buffers=None
        self._buffer_views=None
        self._n_crop_failures=0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    assert torch.cuda.is_available(), 'A CUDA GPU is required'
    store = st.open_store('/orcd/compute/sdorkenw/001/segclr-db', 'microns')
    cache = SkeletonCache(store)
    roots = [int(x) for x in Path(os.environ.get('INFERENCE_ROOTS',str(BASE/'roots.txt'))).read_text().split()]
    available = cache.list_named_skeletons()
    available = available[(available.skeleton_name==NAME) & available.root_id.isin(roots)]
    if args.pilot:
        roots = [int(available.sort_values('n_nodes').iloc[0].root_id)]
    else:
        roots = roots[args.rank::args.world]
    inference = None
    dest_dir = BASE/'named_embeddings'/NAME/RUN/CHECKPOINT
    dest_dir.mkdir(parents=True,exist_ok=True)
    failures=[]
    for rid in roots:
        try:
            skel=cache.get_named_skeleton(rid,NAME)
            if args.pilot:
                inference = NamedInference()
                ids, vectors = inference._embed(rid,skel,np.arange(min(32,len(skel))))
                assert len(ids)==min(32,len(skel)) and np.isfinite(vectors).all()
                # Exercise named storage and readback in a temporary test store only.
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    test=st.init_store(tmp,'microns',datastack='minnie65_phase3_v1',mat_version=1718)
                    write(test,rid,ids,vectors)
                    got=st.scan(test,'named_node_embeddings',dim=vectors.shape[1]).to_pydict()
                    np.testing.assert_array_equal(got['node_id'],ids)
                    np.testing.assert_array_equal(got['embedding'],vectors)
                    assert st.count_rows(test,'node_embeddings',dim=vectors.shape[1])==0
                print(f'PILOT PASSED: {rid}, {len(ids)} nodes, {vectors.shape[1]} dimensions; named-storage roundtrip passed',flush=True)
                (BASE/'inference_ready_roots.txt').write_text(''.join(f'{int(r)}\n' for r in sorted(available.root_id)))
                print(f'Inference-ready cohort: {len(available)} cells',flush=True)
                return
            dest=dest_dir/f'{rid}.npz'
            with (dest_dir/f'{rid}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if dest.exists():
                    with np.load(dest) as data:
                        ids,vectors=data['node_ids'],data['embeddings']
                        assert str(data['skeleton_name'])==NAME
                        assert int(data['root_id'])==rid
                        assert str(data['run_id'])==RUN and str(data['checkpoint_id'])==CHECKPOINT
                        assert vectors.shape==(len(skel),64) and np.isfinite(vectors).all()
                else:
                    if inference is None:
                        inference = NamedInference()
                    ids,vectors=inference._embed(rid,skel,np.arange(len(skel)))
                    assert len(ids)==len(skel) and np.isfinite(vectors).all()
                    partial=dest.with_suffix('.partial.npz')
                    np.savez_compressed(partial,root_id=rid,skeleton_name=NAME,
                                        run_id=RUN,checkpoint_id=CHECKPOINT,
                                        node_ids=ids,embeddings=vectors)
                    os.replace(partial,dest)
                np.testing.assert_array_equal(ids,np.arange(len(skel)))
                # Serialize commits; inference itself runs independently on GPUs.
                with COMMIT_LOCK.open('a') as commit:
                    fcntl.flock(commit,fcntl.LOCK_EX)
                    clause=(f'root_id = {rid} AND skeleton_name = {st.sql_literal(NAME)} '
                            f'AND run_id = {st.sql_literal(RUN)} AND checkpoint_id = {st.sql_literal(CHECKPOINT)}')
                    old=st.scan(store,'named_node_embeddings',dim=vectors.shape[1],filter=clause,
                                columns=['node_id']).to_pydict()['node_id']
                    if old:
                        np.testing.assert_array_equal(np.sort(old),ids)
                    else:
                        write(store,rid,ids,vectors)
                    got=st.scan(store,'named_node_embeddings',dim=vectors.shape[1],filter=clause,
                                columns=['node_id','embedding']).to_pandas().sort_values('node_id')
                    np.testing.assert_array_equal(got.node_id.to_numpy(),ids)
                    np.testing.assert_array_equal(np.stack(got.embedding),vectors)
                print(f'VERIFIED {rid}: {len(ids)} named embeddings',flush=True)
        except Exception as exc:
            logging.exception('Cell %s failed',rid)
            failures.append(dict(root_id=rid,error=str(exc)))
    status_prefix = os.environ.get('INFERENCE_STATUS_PREFIX', 'status')
    (dest_dir/f'{status_prefix}_{args.rank}.json').write_text(json.dumps(dict(cells=len(roots),failures=failures),indent=2))
    if failures: raise RuntimeError(f'{len(failures)} cells failed')


def write(store,rid,ids,vectors):
    n=len(ids)
    st.append(store,'named_node_embeddings',dict(root_id=[rid]*n,skeleton_name=[NAME]*n,
              node_id=ids,run_id=[RUN]*n,checkpoint_id=[CHECKPOINT]*n,
              embedding=vectors.tolist()),dim=vectors.shape[1])


if __name__=='__main__': main()
