"""Execute native notebooks and verify their populations against the database."""
import json
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent
DB=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/native')


def main():
    metadata=json.loads((DB/'metadata.json').read_text())
    assert metadata['n_cells']==2209
    for name in ('skeleton_stats.ipynb','presynaptic_axon_database_figures.ipynb'):
        path=HERE/name
        nb=nbformat.read(path,as_version=4)
        for cell in nb.cells:
            if cell.cell_type=='code':
                cell.outputs=[]
                cell.execution_count=None
        print('Executing',path,flush=True)
        client=NotebookClient(nb,timeout=3600,kernel_name='python3',
                             resources={'metadata':{'path':str(HERE)}})
        client.execute()
        assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
        figures=sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)
        has_component_diagnostics = any(
            c.cell_type == 'markdown' and c.source.startswith('## Presynaptic sites per connected component')
            for c in nb.cells)
        expected_figures = (7 if name == 'skeleton_stats.ipynb' else 3) + 2 * has_component_diagnostics
        assert figures == expected_figures, figures
        temp=path.with_suffix('.executed.tmp.ipynb')
        nbformat.write(nb,temp)
        os.replace(temp,path)
        print(f'Saved {name}: {figures} embedded figures',flush=True)
    completed=list((HERE/'components_gt5_presynaptic/cache').glob('*/complete.json'))
    assert len(completed)==1
    cached=json.loads(completed[0].read_text())
    assert cached['cells']==2209
    assert cached['kept_edges']==metadata['totals']['retained_edges']
    report=dict(n_cells=metadata['n_cells'],totals=metadata['totals'],
                cached_edges=cached['kept_edges'],notebooks=['skeleton_stats.ipynb','presynaptic_axon_database_figures.ipynb'])
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
