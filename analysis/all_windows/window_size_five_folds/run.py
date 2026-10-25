"""Prepare shared CPU caches or run one missing window-size/fold training job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pack', 'train'])
    parser.add_argument('index', type=int)
    cli = parser.parse_args()
    plan = json.loads((BASE / 'plan.json').read_text())
    item = plan['packing' if cli.mode == 'pack' else 'jobs'][cli.index]
    config = ROOT / item['config']
    os.chdir(ROOT)
    if cli.mode == 'train':
        report = ROOT / item['report']
        if report.exists() and 'window_test_metrics' in json.loads(report.read_text()):
            print(f'Already complete: {report}', flush=True)
            return
        import torch
        assert torch.cuda.is_available(), 'GPU required; refusing CPU fallback'
        env = os.environ.copy()
        env['SEGCLR_PACKING_CACHE_ONLY'] = '1'
        env['SEGCLR_PREDICTION_CACHE'] = str(Path(plan['prediction_root']) / f'task{cli.index}')
        child = subprocess.Popen([sys.executable, '-u', 'scripts/train_gnn.py', '--config', str(config)],
                                 env=env, start_new_session=True)
        def stop(signum, frame):
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            raise SystemExit(99)
        signal.signal(signal.SIGUSR1, stop)
        signal.signal(signal.SIGTERM, stop)
        code = child.wait()
        if code or not report.exists():
            raise SystemExit(code or 1)
        return
    import torch
    from scripts.train_gnn import build_training_parser, build_training_datasets
    from scripts.training_config import parse_training_args
    from data.packing_cache import identity, prepare_cached, ready
    from data.packed_windows import PackedWholeCellWindows
    torch.set_num_threads(1)
    args = parse_training_args(build_training_parser(), ['--config', str(config)])
    manifest = json.loads(Path(args.manifest).read_text())
    key = identity(args, manifest)
    assert key == item['key'], 'Packing inputs changed since plan creation'
    if ready(key):
        print(f'Already packed: {key}', flush=True)
        return
    train, test, _ = build_training_datasets(args)
    for split, source in [('train', train), ('test', test)]:
        dataset = PackedWholeCellWindows(source)
        original = dataset.prepare_packed
        def prepare(*, _original=original, **kwargs):
            kwargs['max_bytes'] = 256 * 2**30
            return _original(**kwargs)
        dataset.prepare_packed = prepare
        assert prepare_cached(dataset, key, split, workers=15, build=True)
        del dataset
    print(f'Packing complete: {key}', flush=True)


if __name__ == '__main__':
    main()
