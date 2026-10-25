"""Read legacy inline or shared postsynaptic feature arrays."""
from pathlib import Path

import numpy as np


def load_postsynaptic_features(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        if 'post_x' in cache:
            return cache['post_x'].copy()
        feature_path = path.parent / str(cache['feature_path'])
        with np.load(feature_path, allow_pickle=False) as features:
            vectors = features['post_x']
            rows = cache['feature_rows']
            if rows.dtype.kind not in 'iu' or np.any(rows < 0) or np.any(rows >= len(vectors)):
                raise ValueError(f'Invalid postsynaptic feature alignment: {path}')
            return vectors[rows].copy()
