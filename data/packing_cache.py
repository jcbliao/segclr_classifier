"""Atomic, locked NumPy packing caches shared by CPU builders and GPU trainers."""
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import numpy as np
from data.packed_segmentation_augmentation import enabled as segmentation_enabled
from data.segmentation_augmentation import TRAINING_POLICIES

DEFAULT_ROOT = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/packed_windows')
ROOT = Path(__file__).resolve().parents[1]
INPUT_ARGS = ('dataset', 'presynaptic_database', 'postsynaptic_cache', 'use_postsynaptic',
              'single_presynaptic_embedding', 'presynaptic_compartment_filter',
              'num_embeddings', 'window_nm', 'neighborhood_root', 'gt_pos_dim',
              'gt_use_thickness')

def identity(args, manifest):
    config = {name: getattr(args, name, None) for name in INPUT_ARGS}
    config['geometry'] = (args.dataset == 'all_windows' or args.spatial or args.position
                          or args.lpe or args.architecture not in ('mean', 'pointwise_mlp', 'linear'))
    # Ignore experiment bookkeeping: only the hierarchy and exact cell/split definitions affect packing.
    config['manifest'] = {k:v for k,v in manifest.items() if k not in ('description','created_at','fold','name')}
    paths = []
    for rid in manifest['cells']:
        if args.dataset == 'all_windows':
            paths += [ROOT/'data/graph_cache'/f'{rid}.pt',
                      Path(args.neighborhood_root)/'neighborhoods'/f'n{args.num_embeddings}'/f'{rid}.npz']
            if args.gt_use_thickness: paths.append(ROOT/'data/dendrite_thickness_cache'/f'{rid}.npz')
        else:
            paths.append(Path(args.presynaptic_database)/'cells'/f'{rid}.npz')
    if args.postsynaptic_cache:
        p=Path(args.postsynaptic_cache)
        paths.extend(sorted(p.rglob('*')) if p.is_dir() else [p])
        marker = p/'complete.json'
        if marker.exists():
            shared = json.loads(marker.read_text()).get('shared_feature_store')
            if shared:
                paths.extend(sorted((p/shared).resolve().rglob('*')))
    if getattr(args,'embedding_augmentation',None) in TRAINING_POLICIES:
        config['segmentation_augmentation_database']=str(Path(args.embedding_augmentation_database).resolve())
        paths.extend(Path(args.embedding_augmentation_database)/'segmask'/'cells'/f'{rid}.npz' for rid in manifest['cells'])
    config['sources'] = [(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.is_file()]
    modules=['packing_cache.py','postsynaptic_cache.py','packed_windows.py','packed_presynaptic.py','batched_presynaptic.py',
             'dataset_windowed.py','dataset_presynaptic.py','geodesic_window.py','dataset_lcpn.py',
             'packed_segmentation_augmentation.py','segmentation_augmentation.py']
    config['code']={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() for name in modules}
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

def cache_dir(key, split):
    return Path(os.environ.get('SEGCLR_PACKING_CACHE',DEFAULT_ROOT))/key/split

def ready(key):
    return all((cache_dir(key,s)/'complete.json').exists() for s in ('train','test'))

_training_leases = ContextVar('packing_training_leases', default=None)
_training_mappings = ContextVar('packing_training_mappings', default=None)

def _lease(key):
    if not key or Path(key).name != key or key in ('.', '..'):
        raise ValueError('Packing cache key must be a single directory name')
    # Keep lock inodes outside the cache directories that cleanup removes.
    directory = cache_dir(key, 'train').parent.parent / '.leases'
    directory.mkdir(parents=True, exist_ok=True)
    # NFS implements shared flock using a read lock, requiring a readable fd.
    lock = (directory / f'{key}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_SH)
    return lock

@contextmanager
def training_cache_session():
    """Keep caches through a run; the last successful consumer removes them."""
    leases = {}
    token = _training_leases.set(leases)
    mappings = []
    mappings_token = _training_mappings.set(mappings)
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        _training_leases.reset(token)
        _training_mappings.reset(mappings_token)
        # Unlinking an open mmap on NFS creates .nfs files and prevents rmdir.
        # All consumers in this training session have finished at this point.
        for mapping in mappings:
            if not mapping._mmap.closed:
                mapping._mmap.close()
        for key, lock in leases.items():
            fcntl.flock(lock, fcntl.LOCK_UN)
            try:
                if succeeded:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue  # Another trainer or CPU builder still needs it.
                    directory = cache_dir(key, 'train').parent
                    if directory.is_symlink():
                        raise ValueError('Refusing to remove a symlinked packing cache')
                    if directory.exists():
                        shutil.rmtree(directory)
                        Path(lock.name).with_suffix('.cleaned').touch()
                        print(f'Removed completed training cache: {directory}', flush=True)
            finally:
                lock.close()

def _cache_access(function):
    @wraps(function)
    def wrapped(dataset, key, *args, **kwargs):
        leases = _training_leases.get()
        if leases is not None:
            if key not in leases:
                leases[key] = _lease(key)
            return function(dataset, key, *args, **kwargs)
        # Standalone CPU builders retain their output for future trainers.
        with _lease(key):
            return function(dataset, key, *args, **kwargs)
    return wrapped

def _load(dataset, directory):
    meta=json.loads((directory/'complete.json').read_text())
    packed={}
    for name,desc in meta['arrays'].items():
        value=None if desc is None else np.load(directory/desc,mmap_mode='c',allow_pickle=False)
        mappings = _training_mappings.get()
        if isinstance(value,np.memmap) and mappings is not None:
            mappings.append(value)
        if name.startswith('fields/'):
            packed.setdefault('fields',{})[name.split('/')[1]]=value
        else: packed[name]=value
    if len(dataset)!=meta['windows']:raise ValueError('Packing cache window count mismatch')
    dataset._packed=packed
    dataset.packing_bytes=meta['bytes'];dataset.packing_seconds=0.0
    print(f'Loaded packed cache: {directory} ({meta["windows"]:,} windows, {meta["bytes"]/2**30:.2f} GiB)',flush=True)
    return True

@_cache_access
def prepare_cached(dataset, key, split, workers=15, build=False):
    if getattr(dataset,'embedding_augmentation',None) is not None and not segmentation_enabled(dataset):return False
    directory=cache_dir(key,split)
    if (directory/'complete.json').exists():return _load(dataset,directory)
    cleaned = directory.parent.parent / '.leases' / f'{key}.cleaned'
    if not build and os.environ.get('SEGCLR_PACKING_CACHE_ONLY')=='1' and not cleaned.exists():
        raise RuntimeError(f'CPU packing cache is not ready: {directory}')
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/f'{split}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'complete.json').exists():return _load(dataset,directory)
        stage=directory.parent/f'.{split}.{uuid.uuid4().hex}'
        stage.mkdir();dataset._packing_workdir=stage
        started=time.monotonic()
        try:
            if not dataset.prepare_packed(workers=workers,max_bytes=44*2**30):return False
            arrays={};total=0
            for name,value in dataset._packed.items():
                values={f'fields/{k}':v for k,v in value.items()} if name=='fields' else {name:value}
                for label,array in values.items():
                    if array is None:arrays[label]=None;continue
                    target=stage/f'{label}.npy';target.parent.mkdir(exist_ok=True)
                    if isinstance(array,np.memmap) and Path(array.filename)==target:array.flush()
                    else:np.save(target,array,allow_pickle=False)
                    arrays[label]=str(target.relative_to(stage));total+=array.nbytes
            meta={'windows':len(dataset),'bytes':total,'arrays':arrays,'seconds':time.monotonic()-started}
            (stage/'complete.json').write_text(json.dumps(meta,indent=2)+'\n')
            os.rename(stage,directory)
        finally:
            del dataset._packing_workdir
            if stage.exists():shutil.rmtree(stage)
    return _load(dataset,directory)
