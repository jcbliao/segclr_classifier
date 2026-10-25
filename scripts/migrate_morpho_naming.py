"""Rename local morphology experiment names without rewriting frozen artifacts.

Old paths remain relative symlinks. JSON/CSV/parquet/checkpoints are deliberately
byte-preserved: their columns, source IDs and hashes are historical provenance.
Run with --apply; without it this prints the planned path renames.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import tokenize
from pathlib import Path


def canonical(value):
    return re.sub('casey', lambda m: {'casey':'morpho','Casey':'Morpho','CASEY':'MORPHO'}.get(m[0],'morpho'), value, flags=re.I)


def paths(root):
    # No scans or mutations of shared datasets, environments or historical logs.
    for name in ('scripts','data','analysis','results','models','docs','configs','archive'):
        folder=root/name
        if not folder.exists():
            continue
        yield from (p for p in folder.rglob('*') if not p.is_symlink()
                    and not any(part.startswith(('logs','__pycache__')) for part in p.relative_to(folder).parts))


def prose(text):
    # Preserve the contributor's real name when explicitly giving attribution.
    return re.sub(r'\bCasey\b(?! Schneider)', 'Morpho', text)


def presentation(text, modules):
    text=prose(text)
    # Canonical local paths and executable names; external storage identifiers stay valid.
    for prefix in ('analysis/','results/','models/','data/','scripts/','presynaptic/'):
        text=re.sub(re.escape(prefix)+r'[^\s\"\'`<>)}\],;]*', lambda m: canonical(m[0]), text)
    for old,new in modules.items():
        text=re.sub(r'(?<![\w])'+re.escape(old)+r'(?![\w])',new,text)
    text=re.sub(r'(#SBATCH\s+--job-name=)(\S+)',lambda m:m[1]+canonical(m[2]),text)
    return text


def python_source(text,modules):
    lines=text.splitlines(keepends=True)
    starts=[0]
    for line in lines:
        starts.append(starts[-1]+len(line))
    edits=[]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        value=token.string
        if token.type in (tokenize.COMMENT,tokenize.STRING):
            value=presentation(value,modules)
        elif token.type==tokenize.NAME and value in modules:
            value=modules[value]
        if value!=token.string:
            start=starts[token.start[0]-1]+token.start[1]
            stop=starts[token.end[0]-1]+token.end[1]
            edits.append((start,stop,value))
    for start,stop,value in reversed(edits):
        text=text[:start]+value+text[stop:]
    return text


def migrate(root,apply=False):
    root=Path(root)
    inventory=list(paths(root))
    named=sorted((p for p in inventory if 'casey' in p.name.lower()),key=lambda p:len(p.parts),reverse=True)
    modules={p.stem:canonical(p.stem) for p in named
             if p.suffix=='.py' or (p.is_dir() and p.name.isidentifier())}
    renames=[]
    for old in named:
        new=old.with_name(canonical(old.name))
        if new.exists() or new.is_symlink():
            raise FileExistsError(f'Cannot overwrite {new}')
        renames.append({'legacy':old.relative_to(root).as_posix(),
                        'canonical':canonical(old.relative_to(root).as_posix())})
        if apply:
            old.rename(new)
            old.symlink_to(new.name,target_is_directory=new.is_dir())
    changed=[]
    if apply:
        for path in paths(root):
            if not path.is_file() or path.suffix not in ('.py','.sh','.md','.ipynb'):
                continue
            # Frozen reference implementations and archived training records remain intact.
            if ('archive' in path.relative_to(root).parts or path.name.startswith('reference_')
                    or path.name in ('migrate_morpho_naming.py','test_morpho_naming.py')):
                continue
            original=path.read_text()
            if path.suffix=='.ipynb':
                notebook=json.loads(original)
                altered=False
                for cell in notebook.get('cells',[]):
                    source=cell.get('source',[])
                    text=''.join(source) if isinstance(source,list) else source
                    updated=(python_source(text,modules) if cell['cell_type']=='code' else presentation(text,modules))
                    if text!=updated:
                        cell['source']=updated.splitlines(keepends=True)
                        altered=True
                    for output in cell.get('outputs',[]):
                        for container,key in [(output,'text')]+[(output.get('data',{}),mime)
                                for mime in ('text/plain','text/html','image/svg+xml')]:
                            value=container.get(key)
                            if not isinstance(value,(str,list)):
                                continue
                            new=([presentation(v,modules) for v in value] if isinstance(value,list)
                                 else presentation(value,modules))
                            if new!=value:
                                container[key]=new
                                altered=True
                if not altered:
                    continue
                updated=json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
            else:
                updated=python_source(original,modules) if path.suffix=='.py' else presentation(original,modules)
            if updated!=original:
                path.write_text(updated)
                changed.append(path.relative_to(root).as_posix())
    return {'renames':renames,'presentation_files_updated':changed,
            'preserved':'Frozen JSON/CSV/parquet/checkpoints, source attribution, external storage and database IDs; old local paths are compatibility symlinks.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    result=migrate(args.root,args.apply)
    if args.apply:
        target=args.root/'docs/morpho_naming_migration.json'
        target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
