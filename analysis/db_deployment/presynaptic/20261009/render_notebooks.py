from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path('/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier')
for name in ['test_analysis','unlabeled_analysis']:
    path=root/'analysis/db_deployment/presynaptic/20261009'/f'{name}.ipynb'
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
    nbformat.write(notebook,path)
    print('Rendered',name,flush=True)
