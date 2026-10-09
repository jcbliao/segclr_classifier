"""Combine compatible per-cell precomputed mesh directories into one mesh source."""
import argparse
import json
from pathlib import Path
import shutil


def plan(source, variant):
    entries = []
    common_info = None
    for cell in sorted(source.iterdir()):
        if not cell.is_dir() or not cell.name.isdigit():
            continue
        mesh = cell / variant
        if not (mesh / 'info').exists():
            continue
        info = json.loads((mesh / 'info').read_text())
        if info.get('@type') != 'neuroglancer_multilod_draco' or info.get('sharding'):
            raise ValueError(f'Unsupported mesh format: {mesh}')
        if common_info is None:
            common_info = info
        elif info != common_info:
            raise ValueError(f'Incompatible mesh metadata: {mesh}')
        files = [mesh / cell.name, mesh / f'{cell.name}.index']
        if not all(p.is_file() and p.stat().st_size > 0 for p in files):
            raise ValueError(f'Missing or empty mesh/index: {mesh}')
        entries.append((cell.name, files))
    if not entries:
        raise ValueError('No compatible cell meshes found')
    return common_info, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--variant', required=True, help='Per-cell mesh directory, e.g. mesh_healed')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    info, entries = plan(args.source, args.variant)
    total = sum(p.stat().st_size for _, files in entries for p in files)
    print(json.dumps(dict(cells=len(entries), bytes=total, root_ids=[root for root, _ in entries]), indent=2))
    if args.dry_run:
        return
    if args.output is None:
        parser.error('--output is required unless --dry-run')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    mesh_output = args.output / 'mesh'
    mesh_output.mkdir(parents=True)
    for _, files in entries:
        for file in files:
            shutil.copy2(file, mesh_output / file.name)
    (mesh_output / 'info').write_text(json.dumps(info, indent=2))
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()), variant=args.variant,
        root_ids=[root for root, _ in entries], bytes=total,
        viewer_source='precomputed://<server URL>/mesh',
        coordinate_units='nm; source transform preserved'), indent=2))
    print(f'Saved combined mesh source: {mesh_output}')


if __name__ == '__main__':
    main()
