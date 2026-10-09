"""Copy a quiescent tree in parallel, verify it, then retain its path as a symlink."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess


def relocate(source, destination, workers, resume=False):
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise ValueError('source points to another destination')
        return
    if not source.is_dir() or (destination.exists() and not resume):
        raise ValueError('source must exist; existing destination requires --resume')
    destination.mkdir(parents=True, exist_ok=resume)
    entries = list(source.iterdir())

    def copy(entry):
        args = ['rsync', '-rlt', '--', str(entry), str(destination) + '/']
        subprocess.run(args, check=True)
        print('copied', entry.name, flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy, entries))
    # rsync verifies transferred file data internally. This second pass checks
    # that every source path, size, mtime and link still agrees before cutover.
    check = subprocess.run(['rsync', '-rltn', '--omit-dir-times', '--delete', '--out-format=%n', '--',
                            str(source) + '/', str(destination) + '/'],
                           check=True, text=True, capture_output=True)
    if check.stdout.strip():
        raise RuntimeError('tree changed or copy is incomplete: ' + check.stdout[:4000])
    backup = source.with_name(source.name + '.relocation_backup')
    if backup.exists():
        raise ValueError('backup already exists')
    os.rename(source, backup)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except BaseException:
        os.rename(backup, source)
        raise
    print('verified and switched', source, '->', destination, flush=True)
    shutil.rmtree(backup)
    print('released old scratch copy', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    relocate(args.source, args.destination, args.workers, args.resume)
